import pdb
import argparse
import json
from typing_extensions import Annotated, TypedDict
import ast
import signal
import os

from langchain import hub
from langchain_core.documents.base import Document
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from database.participant_pg_database import DbConfig, load_participant_db, format_query_result

from dotenv import load_dotenv
load_dotenv()

# Default configuration
def DEFAULT_CONFIG():
    return {
        "use_vector_search": True,
        "vector_search_k": 10,
        "use_llm_for_column_selection": False,
        "use_mock_data": False
    }

# Parse configuration from command-line flags
def hybrid_chatbot_parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--config_file_json',
        type=str,
        help='Path to the JSON config file',
        default=None)
    
    return parser.parse_args()

def extract_selected_columns(query: str):
    lower_query = query.lower()
    if "select *" in lower_query:
        return []
    start_idx = lower_query.index("select") + len("select")
    end_idx = lower_query.index("from")
    columns_part = query[start_idx:end_idx].strip()
    return [col.strip() for col in columns_part.split(",")]

class SQLGenOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

text_to_sql_tmpl = """\
Given an input question, first create a syntactically correct {dialect} 
query to run, then look at the results of the query and return the answer. 
You can order the results by a relevant column to return the most 
interesting examples in the database.

Pay attention to use only the column names that you can see in the schema 
description. Be careful to not query for columns that do not exist. 
Pay attention to which column is in which table. Also, qualify column names 
with the table name when needed. 

IMPORTANT NOTE: you can use specialized pgvector syntax (`<=>`) to do nearest 
neighbors/semantic search to a given vector from an embeddings column in the table. 
The embeddings value for a given row typically represents the semantic meaning of that row. 
The vector represents an embedding representation 
of the question, given below. Do NOT fill in the vector values directly, but rather specify a 
`[query_vector]` placeholder. For instance, some select statement examples below 
(the name of the embeddings columns columns are like `column_name_embedding`):
SELECT * FROM items ORDER BY Languages_embedding <=> '[query_vector]' LIMIT 5;
SELECT * FROM items WHERE id != 1 ORDER BY Languages_embedding <=> (SELECT Languages_embedding FROM items WHERE id = 1) LIMIT 5;
SELECT * FROM items WHERE Skills_embedding <=> '[query_vector]' < 5;
Use this vector search always instead of the LIKE or ILIKE operator. Never use LIKE or ILIKE.

You are required to use the following format, 
each taking one line:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

Only use tables listed below.
{schema}


Question: {input}
SQLQuery: \
"""

class ChatbotPipeline:
    def __init__(self, config, participant_db):
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o")
        
        self.participant_db = participant_db

        self.structured_cols = self.participant_db.get_structured_table_column_names()
        self.unstructured_cols = self.participant_db.get_unstructured_table_column_names()
        self.query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
        self.vector_sql_prompt_template = ChatPromptTemplate.from_template(text_to_sql_tmpl)
        self.embedding_model = OpenAIEmbeddings()

    def identify_columns(self, query, columns):
        prompt = f"""
        Identify the columns in the database schema that are relevant to the following user query:

        Query: "{query}"

        Available Columns: {columns}

        Respond  with a comma-separated list of column names.
        """
        response = self.llm.invoke(prompt)
        columns_list = response.content.split(',')
        columns_list = [col.strip() for col in columns_list]
        return list(filter(lambda col: col in columns, columns_list))
    
    def write_query(self, state: State):
        """Generate SQL query to fetch information."""
        prompt = self.vector_sql_prompt_template.invoke(
            {
                "dialect": self.participant_db.lc_db.dialect,
                "schema": self.participant_db.lc_db.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = self.llm.with_structured_output(SQLGenOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}


    def execute_query(self, state: State):
        """Execute SQL query."""
        pdb.set_trace()
        return {"result": self.participant_db.lc_db_query_tool.invoke(state["query"])}

    def create_semantic_entities(self, user_query):
        prompt = f"""
        Your task is to examine the user query: "{user_query}" and determine which entities belong to each of the following categories:

        {self.unstructured_cols}

        Instructions:
        1. Output only valid JSON (no additional text or commentary).
        2. Each category (column name) is a top-level key in the JSON object.
        3. For each category, provide a list of extracted phrases as its value.
        4. If no phrases match a category, you may omit that category or set its value to an empty list.
        5. Do not repeat or include the example JSON in your final output. The example below is for reference only.

        Example JSON schema (for reference only; do not include it verbatim in your answer):
        {{
        "skills": ["python", "machine learning"],
        "education_specialization": ["computer science"],
        "past_jobs": ["software engineer at ABC"]
        }}

        Final Answer Requirements:
        - Return only the JSON object containing categories as keys and arrays of extracted phrases as values.
        - No extra text or formatting outside the JSON.
        """
        structured_llm = self.llm.with_structured_output(self.participant_db.pydantic_unstructured_categories)
        unstructured_cat_response = structured_llm.invoke(prompt)
        print(f"Semantic Entities: \n {unstructured_cat_response}")
        semantic_entities_dict = unstructured_cat_response.dict()
        return semantic_entities_dict

    def execute_semantic_queries(self, semantic_entities_dict):
        print("Processing semantic queries")
        entities = []
        for category, c_entities in semantic_entities_dict.items():
            for entity in c_entities:
                entities.append(category + ": " + entity)

        semantic_results = []
        if self.config["use_vector_search"]:
            print("Processing by vector search")
            all_faiss_results = []
            seen_ids = set()
            for entity in entities:
                faiss_results = self.faiss_store.similarity_search(entity, k=self.config["vector_search_k"])
                for result in faiss_results:
                    sp_id = result.metadata["id"]
                    if sp_id not in seen_ids:
                        seen_ids.add(sp_id)
                        all_faiss_results.append((sp_id, result.metadata["text"]))
            # TODO: DEDUPLICATE RETRIEVALS
            if all_faiss_results:
                pretty_result = format_query_result(all_faiss_results, headers=["SP ID", "Text"])
                semantic_results.append(pretty_result)
        if self.config["use_llm_for_column_selection"]:
            for column, items in semantic_entities_dict.items():
                if items:
                    query = f"SELECT sp_id, {column} FROM participants_unstructured"
                    response = self.participant_db.execute_unstructured_query_tool.invoke(query)
                    parsed_response = ast.literal_eval(response)
                    pretty_response = format_query_result(parsed_response, headers=["sp_id", column])
                    semantic_results.append(pretty_response)
        print("Semantic Query processed.")
        return semantic_results

    def process_results_with_llm(self, sql_query, sql_results, semantic_results, identified_cols, user_query):
        print("Processing combined structured and semantic results in one.")
        # Make SQL Results into a pretty string and add headers from the SQL query
        pretty_sql_results = "No structured data found."
        if sql_results["result"]:
            parsed_sql_result = ast.literal_eval(sql_results["result"])    
            headers = extract_selected_columns(sql_query["query"])
            if headers and len(parsed_sql_result) > 0 and len(parsed_sql_result[0]) != len(headers):
                print("Warning: number of columns does not match extracted headers. Headers ignored.")
                headers = None
            pretty_sql_results = format_query_result(parsed_sql_result, headers=headers)
        # Make Semantic Results into a pretty string
        semantic_results_string = ""
        for result in semantic_results:
            content = result.page_content if isinstance(result, Document) else result
            semantic_results_string += content + "\n"
        prompt = f"""
Based on the user's query: "{user_query}", answer the query by analyzing and aggregating the following data:

The following categories/ columns were identified in the user query: {identified_cols}

SQL processed results for structured columns related to the identified categories:
{pretty_sql_results}

Created by this SQL Query:
{sql_query['query']}

Semantically retrieved entities for unstructured columns related to the identified categories:
{semantic_results_string}
        """
        print("Combining Prompt: \n")
        print(prompt)
        response = self.llm.invoke(prompt)
        return response.content, prompt

    def chatbot(self, query):
        identified_cols = self.identify_columns(query, columns=self.structured_cols + self.unstructured_cols)
        # identified_structured_cols = [col for col in identified_cols if col in self.structured_cols]
        # identified_unstructured_cols = [col for col in identified_cols if col in self.unstructured_cols]
        sql_query = self.write_query({"question": query})
        sql_response = self.execute_query({"query": sql_query["query"]})
        pdb.set_trace()
        semantic_results = []        
        final_response, prompt = self.process_results_with_llm(sql_query, sql_response, semantic_results, identified_cols, query)
        return final_response, prompt

def handle_exit(signal_received, frame):
    print("\n[INFO] Chatbot exiting... Conversation saved.")
    exit(0)  # Ensures clean exit

# Register signal handler for `Ctrl+C`
signal.signal(signal.SIGINT, handle_exit)

if __name__ == "__main__":

    args = hybrid_chatbot_parse_args()
    if args.config_file_json:
        with open(args.config_file_json, 'r') as config_file:
            config = json.load(config_file)
    else:
        config = DEFAULT_CONFIG()
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    db_config = DbConfig(
        os.getenv("DB_USER"),
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        os.getenv("DB_NAME"),
        os.getenv("DB_PASSWORD"),
        os.getenv("OPENAI_API_KEY")
    )

    participant_db = load_participant_db(db_config)
    pipeline = ChatbotPipeline(config, participant_db)
    print("Chatbot is running. Type your query below (or type 'exit' to quit):")
    while True:
        print("\n")
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        response, prompt = pipeline.chatbot(user_input)
        log_file = "chatbot/chatbot_conversation_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"You: \n{user_input}\n\n")
            f.write(f"Prompt: \n{prompt}\n\n")
            f.write(f"Bot: \n{response}\n\n\n")
        print(f"Chatbot: {response}")

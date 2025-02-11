import argparse
import json
from typing_extensions import Annotated
from typing_extensions import TypedDict
import ast
import signal

from langchain import hub
from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from preprocessing.participant_database import create_mock_participant_database, create_participant_database, format_query_result

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

class ChatbotPipeline:
    def __init__(self, config):
        self.config = config
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        
        self.participant_database = None
        if self.config["use_mock_data"]:
            self.participant_database = create_mock_participant_database(
                                            structured_db_file = 'sqlite:///chatbot/data/participants_structured_mock.db',
                                            unstructured_db_file = 'sqlite:///chatbot/data/participants_unstructured_mock.db'
                                            )
        else:
            self.participant_database = create_participant_database(
                                            structured_db_file = "sqlite:///chatbot/data/participants_structured.db",
                                            unstructured_db_file = "sqlite:///chatbot/data/participants_unstructured.db")

        self.structured_cols = self.participant_database.get_structured_column_names()
        self.unstructured_cols = self.participant_database.get_unstructured_column_names()
        
        self.query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

        self.embedding_model = OpenAIEmbeddings()
        self.faiss_store = self.participant_database.make_faiss_index(self.embedding_model)

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
        prompt = self.query_prompt_template.invoke(
            {
                "dialect": self.participant_database.sql_db_structured.dialect,
                "top_k": 10000,
                "table_info": self.participant_database.sql_db_structured.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = self.llm.with_structured_output(SQLGenOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}


    def execute_query(self, state: State):
        """Execute SQL query."""
        return {"result": self.participant_database.execute_structured_query_tool.invoke(state["query"])}

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
        structured_llm = self.llm.with_structured_output(self.participant_database.pydantic_unstructured_categories)
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

            if all_faiss_results:
                pretty_result = format_query_result(all_faiss_results, headers=["SP ID", "Text"])
                semantic_results.append(pretty_result)
        if self.config["use_llm_for_column_selection"]:
            for column, items in semantic_entities_dict.items():
                if items:
                    query = f"SELECT sp_id, {column} FROM participants_unstructured"
                    response = self.participant_database.execute_unstructured_query_tool.invoke(query)
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
        identified_structured_cols = [col for col in identified_cols if col in self.structured_cols]
        sql_query = ""
        sql_response = None
        if identified_structured_cols:
            sql_query = self.write_query({"question": query})
            sql_response = self.execute_query({"query": sql_query["query"]})
        semantic_results = []
        identified_unstructured_cols = [col for col in identified_cols if col in self.unstructured_cols]
        if identified_unstructured_cols:
            semantic_entities_dict = self.create_semantic_entities(query)
            semantic_results = self.execute_semantic_queries(semantic_entities_dict)
        
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
    
    print("Chatbot is running. Type your query below (or type 'exit' to quit):")

    pipeline = ChatbotPipeline(config)
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

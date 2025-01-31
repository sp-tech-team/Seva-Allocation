import argparse
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from typing_extensions import TypedDict
from prettytable import PrettyTable
import ast
import pdb

from langchain import hub
from langchain_core.documents.base import Document
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
from sqlalchemy.sql import text

import faiss

from dotenv import load_dotenv
load_dotenv()

# Default configuration
DEFAULT_CONFIG = {
    "use_vector_search": True,
    "vector_search_k": 10,
    "use_llm_for_column_selection": False
}

# Parse configuration from command-line flags
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path to the JSON config file', default=None)
args = parser.parse_args()

if args.config:
    with open(args.config, 'r') as config_file:
        CONFIG = json.load(config_file)
else:
    CONFIG = DEFAULT_CONFIG


def format_query_result(result, headers=None):
    table = PrettyTable()
    if headers:
        table.field_names = headers  # Set column headers
    for row in result:
        table.add_row(row)  # Add each row of data
    return table.get_string()

def extract_selected_columns(query: str):
    lower_query = query.lower()
    if "select *" in lower_query:
        return []
    start_idx = lower_query.index("select") + len("select")
    end_idx = lower_query.index("from")
    columns_part = query[start_idx:end_idx].strip()
    return [col.strip() for col in columns_part.split(",")]

class UnstructuredCategories(BaseModel):
    """
    A user profile model with optional fields.
    If any field is omitted, it defaults to an empty list.
    However, passing 'null' (None) explicitly is still allowed.
    """

    skills: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of the user's skills (default: empty list if omitted)."
    )
    education_specialization: Optional[List[str]] = Field(
        default_factory=list,
        description="The user's educational specializations (default: empty list if omitted)."
    )
    past_jobs: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of the user's past job titles (default: empty list if omitted)."
    )

class SQLGenOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class ChatbotPipeline:
    def __init__(self):
        self.CONFIG = CONFIG
        self.llm = ChatOpenAI(model="gpt-4o-mini")
        self.structured_engine = create_engine('sqlite:///participants_structured.db')
        self.unstructured_engine = create_engine('sqlite:///participants_unstructured.db')
        self.structured_metadata = MetaData()
        self.unstructured_metadata = MetaData()

        self.participants_structured_table = Table(
            'participants_structured', self.structured_metadata,
            Column('sp_id', Integer, primary_key=True),
            Column('age', Integer),
            Column('gender', String),
            Column('years_of_experience', Integer)
        )
        self.participants_unstructured_table = Table(
            'participants_unstructured', self.unstructured_metadata,
            Column('sp_id', Integer, primary_key=True),
            Column('skills', String),
            Column('education_specialization', String),
            Column('past_jobs', String)
        )

        self.structured_metadata.create_all(self.structured_engine)
        self.unstructured_metadata.create_all(self.unstructured_engine)

        structured_data = [
            {"sp_id": 1, "age": 35, "gender": "Male", "years_of_experience": 10},
            {"sp_id": 2, "age": 28, "gender": "Female", "years_of_experience": 5},
            {"sp_id": 3, "age": 40, "gender": "Male", "years_of_experience": 15},
            {"sp_id": 4, "age": 32, "gender": "Female", "years_of_experience": 8},
            {"sp_id": 5, "age": 45, "gender": "Male", "years_of_experience": 20},
            {"sp_id": 6, "age": 29, "gender": "Female", "years_of_experience": 6},
            {"sp_id": 7, "age": 38, "gender": "Male", "years_of_experience": 12},
            {"sp_id": 8, "age": 27, "gender": "Female", "years_of_experience": 4},
            {"sp_id": 9, "age": 36, "gender": "Male", "years_of_experience": 14},
            {"sp_id": 10, "age": 33, "gender": "Female", "years_of_experience": 9}
        ]
        unstructured_data = [
            {"sp_id": 1, "skills": "Backend Development, Python, APIs", "education_specialization": "Computer Science", "past_jobs": "Software Engineer at XYZ"},
            {"sp_id": 2, "skills": "Frontend Development, React, CSS", "education_specialization": "Information Technology", "past_jobs": "UI Developer at ABC"},
            {"sp_id": 3, "skills": "DevOps, CI/CD, Kubernetes", "education_specialization": "Systems Engineering", "past_jobs": "DevOps Engineer at DEF"},
            {"sp_id": 4, "skills": "Data Science, Machine Learning, R", "education_specialization": "Data Science", "past_jobs": "Data Scientist at GHI"},
            {"sp_id": 5, "skills": "Database Management, SQL, Oracle", "education_specialization": "Database Administration", "past_jobs": "DBA at JKL"},
            {"sp_id": 6, "skills": "Mobile Development, Swift, Android", "education_specialization": "Mobile Computing", "past_jobs": "Mobile Developer at MNO"},
            {"sp_id": 7, "skills": "Cybersecurity, Ethical Hacking, Firewalls", "education_specialization": "Cybersecurity", "past_jobs": "Security Analyst at PQR"},
            {"sp_id": 8, "skills": "Cloud Computing, AWS, Azure", "education_specialization": "Cloud Technology", "past_jobs": "Cloud Engineer at STU"},
            {"sp_id": 9, "skills": "AI Research, NLP, TensorFlow", "education_specialization": "Artificial Intelligence", "past_jobs": "AI Specialist at VWX"},
            {"sp_id": 10, "skills": "Project Management, Agile, Scrum", "education_specialization": "Business Administration", "past_jobs": "Project Manager at YZA"}
        ]

        with self.structured_engine.begin() as conn:
            conn.execute(self.participants_structured_table.delete())
            conn.execute(self.participants_structured_table.insert(), structured_data)

        with self.unstructured_engine.begin() as conn:
            conn.execute(self.participants_unstructured_table.delete())
            conn.execute(self.participants_unstructured_table.insert(), unstructured_data)

        self.sql_db_structured = SQLDatabase(self.structured_engine)
        self.sql_db_unstructured = SQLDatabase(self.unstructured_engine)
        self.execute_structured_query_tool = QuerySQLDatabaseTool(db=self.sql_db_structured)
        self.execute_unstructured_query_tool = QuerySQLDatabaseTool(db=self.sql_db_unstructured)
        self.sql_chain = SQLDatabaseChain(llm=self.llm, database=self.sql_db_structured)
        self.query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")

        self.embedding_model = OpenAIEmbeddings()
        self.faiss_index = faiss.IndexFlatL2(len(self.embedding_model.embed_query("sample text")))
        self.faiss_store = FAISS(
            embedding_function=self.embedding_model,
            index=self.faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        for record in unstructured_data:
            combined_text = (
                f"skills: {record['skills']} "
                f"education_specialization: {record['education_specialization']} "
                f"past_jobs: {record['past_jobs']}"
            )
            embedding = self.embedding_model.embed_query(combined_text)
            metadata = {"id": record["sp_id"], "text": combined_text}
            self.faiss_store.add_texts([combined_text], [metadata])

    def identify_columns(self, query, columns):
        prompt = f"""
        Identify the columns in the database schema that are relevant to the following user query:

        Query: "{query}"

        Available Columns: {columns}

        Respond  with a comma-separated list of column names.
        """
        response = self.llm.invoke(prompt)
        #print("Column Identifier: \n")
        print(response.content)
        columns_list = response.content.split(',')
        columns_list = [col.strip() for col in columns_list]
        return list(filter(lambda col: col in columns, columns_list))
    
    def write_query(self, state: State):
        """Generate SQL query to fetch information."""
        prompt = self.query_prompt_template.invoke(
            {
                "dialect": self.sql_db_structured.dialect,
                "top_k": 10000,
                "table_info": self.sql_db_structured.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = self.llm.with_structured_output(SQLGenOutput)
        result = structured_llm.invoke(prompt)
        print("SQL Query: \n")
        print(result["query"])
        return {"query": result["query"]}


    def execute_query(self, state: State):
        """Execute SQL query."""
        return {"result": self.execute_structured_query_tool.invoke(state["query"])}

    def create_semantic_entities(self, user_query, unstructured_cols):
        prompt = f"""
        Your task is to examine the user query: "{user_query}" and determine which entities belong to each of the following categories:

        {unstructured_cols}

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
        structured_llm = self.llm.with_structured_output(UnstructuredCategories)
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
        if self.CONFIG["use_vector_search"]:
            print("Processing by vector search")
            all_faiss_results = []
            for entity in entities:
                faiss_results = self.faiss_store.similarity_search(entity, k=self.CONFIG["vector_search_k"])
                for result in faiss_results:
                    all_faiss_results.append((result.metadata["id"], result.metadata["text"]))

            if all_faiss_results:
                pretty_result = format_query_result(all_faiss_results, headers=["SP ID", "Text"])
                semantic_results.append(pretty_result)
        if self.CONFIG["use_llm_for_column_selection"]:
            for column, items in semantic_entities_dict.items():
                if items:
                    query = f"SELECT sp_id, {column} FROM participants_unstructured"
                    response = self.execute_unstructured_query_tool.invoke(query)
                    parsed_response = ast.literal_eval(response)
                    pretty_response = format_query_result(parsed_response, headers=["sp_id", column])
                    semantic_results.append(pretty_response)
        print("Semantic Query processed.")
        return semantic_results

    def process_results_with_llm(self, sql_query, sql_results, semantic_results, identified_cols, user_query):
        print("Processing combined structured and semantic results in one.")
        # Make SQL Results into a pretty string and add headers from the SQL query
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
        return response.content

    def chatbot(self, query):
        structured_cols = ["age", "gender", "years_of_experience"]
        unstructured_cols = ["skills", "education_specialization", "past_jobs"]
        identified_cols = self.identify_columns(query, columns=structured_cols + unstructured_cols)
        sql_results = []
        identified_structured_cols = [col for col in identified_cols if col in structured_cols]
        sql_query = ""
        sql_response = None
        if identified_structured_cols:
            sql_query = self.write_query({"question": query})
            sql_response = self.execute_query({"query": sql_query["query"]})
            # sql_query = self.create_sql_query(query, structured_cols)
            # sql_results = self.execute_sql_query(sql_query)
        semantic_results = []
        identified_unstructured_cols = [col for col in identified_cols if col in unstructured_cols]
        if identified_unstructured_cols:
            semantic_entities_dict = self.create_semantic_entities(query, unstructured_cols)
            semantic_results = self.execute_semantic_queries(semantic_entities_dict)
        
        final_response = self.process_results_with_llm(sql_query, sql_response, semantic_results, identified_cols, query)
        return final_response


if __name__ == "__main__":
    print("Chatbot is running. Type your query below (or type 'exit' to quit):")
    pipeline = ChatbotPipeline()
    while True:
        print("\n")
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        response = pipeline.chatbot(user_input)
        print(f"Chatbot: {response}")

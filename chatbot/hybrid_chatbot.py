import argparse
import json
import pdb

from langchain_core.documents.base import Document
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from langchain_openai import OpenAIEmbeddings, OpenAI
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

# Step 1: Set up the SQL database
# Use SQLite for local testing
engine = create_engine('sqlite:///participants.db')
metadata = MetaData()

# Define the participants table
participants_table = Table(
    'participants', metadata,
    Column('sp_id', Integer, primary_key=True),
    Column('age', Integer),
    Column('gender', String),
    Column('years_of_experience', Integer),
    Column('skills', String),
    Column('education_specialization', String),
    Column('past_jobs', String)
)

# Create the table
metadata.create_all(engine)

participants_data = [
    {"sp_id": 1, "age": 35, "gender": "Male", "years_of_experience": 10, "skills": "Backend Development, Python, APIs", "education_specialization": "Computer Science", "past_jobs": "Software Engineer at XYZ"},
    {"sp_id": 2, "age": 28, "gender": "Female", "years_of_experience": 5, "skills": "Frontend Development, React, CSS", "education_specialization": "Information Technology", "past_jobs": "UI Developer at ABC"},
    {"sp_id": 3, "age": 40, "gender": "Male", "years_of_experience": 15, "skills": "DevOps, CI/CD, Kubernetes", "education_specialization": "Systems Engineering", "past_jobs": "DevOps Engineer at DEF"},
    {"sp_id": 4, "age": 32, "gender": "Female", "years_of_experience": 8, "skills": "Data Science, Machine Learning, R", "education_specialization": "Data Science", "past_jobs": "Data Scientist at GHI"},
    {"sp_id": 5, "age": 45, "gender": "Male", "years_of_experience": 20, "skills": "Database Management, SQL, Oracle", "education_specialization": "Database Administration", "past_jobs": "DBA at JKL"},
    {"sp_id": 6, "age": 29, "gender": "Female", "years_of_experience": 6, "skills": "Mobile Development, Swift, Android", "education_specialization": "Mobile Computing", "past_jobs": "Mobile Developer at MNO"},
    {"sp_id": 7, "age": 38, "gender": "Male", "years_of_experience": 12, "skills": "Cybersecurity, Ethical Hacking, Firewalls", "education_specialization": "Cybersecurity", "past_jobs": "Security Analyst at PQR"},
    {"sp_id": 8, "age": 27, "gender": "Female", "years_of_experience": 4, "skills": "Cloud Computing, AWS, Azure", "education_specialization": "Cloud Technology", "past_jobs": "Cloud Engineer at STU"},
    {"sp_id": 9, "age": 36, "gender": "Male", "years_of_experience": 14, "skills": "AI Research, NLP, TensorFlow", "education_specialization": "Artificial Intelligence", "past_jobs": "AI Specialist at VWX"},
    {"sp_id": 10, "age": 33, "gender": "Female", "years_of_experience": 9, "skills": "Project Management, Agile, Scrum", "education_specialization": "Business Administration", "past_jobs": "Project Manager at YZA"}
]

with engine.begin() as conn:
    conn.execute(participants_table.delete())
    conn.execute(participants_table.insert(), participants_data)

sql_db = SQLDatabase(engine)
sql_chain = SQLDatabaseChain(llm=OpenAI(), database=sql_db)

# Step 2: Set up the FAISS vector store
# Initialize FAISS index for semantic search
embedding_model = OpenAIEmbeddings()
sample_embedding = embedding_model.embed_query("sample text")
embedding_dimension = len(sample_embedding)
faiss_index = faiss.IndexFlatL2(embedding_dimension)
faiss_store = FAISS(
    embedding_function=embedding_model,
    index=faiss_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

# Add unstructured data to FAISS for semantic search
unstructured_data = []
for participant in participants_data:
    unstructured_text = "skills: " + participant["skills"] + " education_specialization: " + \
        participant["education_specialization"] + " past_jobs: " + participant["past_jobs"]
    unstructured_data.append({"id": participant["sp_id"], "text": unstructured_text})

for record in unstructured_data:
    embedding = embedding_model.embed_query(record["text"])
    metadata = {"id": record["id"], "text": record["text"]}
    faiss_store.add_texts([record["text"]], [metadata])

def identify_columns(query, columns):
    llm = OpenAI()
    prompt = f"""
    Identify the columns in the database schema that are relevant to the following user query:

    Query: "{query}"

    Available Columns: {columns}

    Respond  with a comma-separated list of column names.
    """
    response = llm(prompt)
    print("Column Identifier: \n")
    print(response)
    columns_list = response.split(',')
    columns_list = [col.strip() for col in columns_list]
    return list(filter(lambda col: col in columns, columns_list))


def create_sql_query(user_query, structured_cols):
    llm = OpenAI()
    prompt = f"""
    Based on the user query: "{user_query}", generate a valid SQL query that retrieves data from the database based on the following schema:
    
    Schema:
    {json.dumps({
        "participants": {
            "columns": structured_cols
        }
    }, indent=2)}

    Rules:
    - The SQL query should only retrieve data from the columns specified in the schema.
    - The SQL query should be valid and executable on the database.
    - Return an empty string if no structured components are found in the query.

    SQL Query:
    """
    response = llm(prompt)
    return response.strip() 


def create_semantic_entities(user_query, unstructured_cols):
    llm = OpenAI()
    prompt = f"""
    Your task is to examine the user query: "{user_query}" and determine which entities belong to each of the following categories:
    {unstructured_cols}

    Instructions:
    1. Return a JSON object where each category (column name) is a key.
    2. Each key has a list of extracted phrases as its value.
    3. If no phrases match a category, either omit that category or provide an empty list.
    4. Include an example JSON schema so it's clear how the output should look.

    Example JSON schema:
    {{
        "skills": ["python", "machine learning"],
        "education_specialization": ["computer science"],
        "past_jobs": ["software engineer at ABC"]
    }}

    Final JSON response:
    """
    response = llm(prompt)
    return response.strip().split(',')


# Step 3: Parse user queries
# Function to process user input and divide into SQL and semantic components
def parse_query(user_query, structured_cols, unstructured_cols):
    llm = OpenAI()
    prompt = f"""
    Your role is to read a user query and create a JSON response that identifies the structured
    and unstructured components of the query based on the following categories:
    Identify the structured and unstructured components of the user query based on the following categories:
    - structured components(SQL-like) components: related to categories {structured_cols}.
    - unstructured (semantic search) components: related to categories {unstructured_cols}.

    Query: "{user_query}"
    
    Respond in JSON format with two keys:
    - sql_conditions: A natural language description of the structured query.
    - semantic_conditions: A list of keywords or phrases for semantic search (e.g., "backend software engineering").

    Extra Rules:
    - Do not parse components that are not part of one of the corresponding categories. Add a blank string to the
      json entry if no related components are found.
    - Only create one json 

    """

    response = llm(prompt)
    print("query Parser: \n")
    print(response)
    try:
        parsed_query = json.loads(response)
        return parsed_query.get("sql_conditions", ""), parsed_query.get("semantic_conditions", [])
    except json.JSONDecodeError:
        raise ValueError("Failed to parse the query using LLM.")

# Step 4: Execute SQL Queries using Text-to-SQL
# Generates and executes SQL queries for complex operations
def execute_sql_query(sql_description):
    if not sql_description:
        return []

    prompt = f"""
    Generate a valid SQL query based on the following database schema and user request:

    Schema:
    {json.dumps({
        "participants": {
            "columns": [
                "sp_id", "age", "gender", "years_of_experience"
            ]
        }
    }, indent=2)}

    User Request: {sql_description}

    SQL Query:
    """
    
    llm = OpenAI()
    response = llm(prompt)
    sql_query = response.strip()
    print(f"Generated SQL Query: \n{sql_query}")
    try:
        with engine.begin() as conn:
            results = conn.execute(text(sql_query)).fetchall()
        return [dict(r._mapping) for r in results]
    except Exception as e:
        print(f"Error executing SQL Query: {e}")
        return []

# Step 5: Execute Semantic Queries
def execute_semantic_queries(semantic_conditions):
    print("Processing semantic queries")
    semantic_results = []
    if CONFIG["use_vector_search"]:
        print("Processing by vector search")
        for condition in semantic_conditions:
            semantic_results.extend(faiss_store.similarity_search(condition, k=CONFIG["vector_search_k"]))
    if CONFIG["use_llm_for_column_selection"]:
        print("Processing by LLM")
        print()
        llm = OpenAI()
        prompt = f"""
        Based on the user's query, decide which columns from the database may contain relevant unstructured data.

        Query: {semantic_conditions}
        Available Columns: ["skills", "education_specialization", "past_jobs"]

        Respond in JSON format with a single key "relevant_columns", whose value is a list of column names.
        """
        response = llm(prompt)
        try:
            parsed_response = json.loads(response)
            relevant_columns = parsed_response.get("relevant_columns", [])
        except json.JSONDecodeError:
            print("Error parsing LLM response for column selection. Defaulting to an empty list.")
            relevant_columns = []
        with engine.connect() as conn:
            for column in relevant_columns:
                query = f"SELECT sp_id, {column} FROM participants"
                semantic_results.extend(conn.execute(query).fetchall())
    print("Semantic Query processed.")
    return semantic_results

# Step 6: Post-process results with LLM
def process_results_with_llm(sql_results, semantic_results, user_query):
    print("Processing combined structured and semantic results in one.")
    llm = OpenAI()
    # Process semantic results safely
    formatted_semantic_results = [
        result.page_content if isinstance(result, Document) else str(result)
        for result in semantic_results
    ]
    print("SQL Result: \n")
    print(sql_results)
    print("Semantic results")
    print(formatted_semantic_results)
    pdb.set_trace()
    prompt = f"""
    Based on the user's query: "{user_query}", answer the query by analyzing and aggregating the following data:

    Retrieved Structured columns:
    {json.dumps(sql_results, default=str)}

    Retrieved Semantic columns:
    {json.dumps(formatted_semantic_results, default=str)}
    """
    
    response = llm(prompt)
    return response

# Step 7: Main Chatbot Functionality
def chatbot(query):
    structured_cols = ["age", "gender", "years_of_experience"]
    unstructured_cols = ["skills", "education_specialization", "past_jobs"]
    identified_cols = identify_columns(query, columns=structured_cols + unstructured_cols)
    sql_description, semantic_conditions = parse_query(query, structured_cols, unstructured_cols)
    sql_results = []
    if any(col in structured_cols for col in identified_cols):
        sql_results = execute_sql_query(sql_description)
    semantic_results = []
    if any(col in unstructured_cols for col in identified_cols):
        semantic_results = execute_semantic_queries(semantic_conditions)
    
    final_response = process_results_with_llm(sql_results, semantic_results, query)
    return final_response


# Main entry point
if __name__ == "__main__":
    print("Chatbot is running. Type your query below (or type 'exit' to quit):")
    while True:
        print("\n")
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        response = chatbot_structured(user_input)
        print(f"Chatbot: {response}")

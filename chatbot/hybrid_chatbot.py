import argparse
import json
from langchain.chains import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData
import faiss

# Default configuration
DEFAULT_CONFIG = {
    "use_vector_search": True,
    "vector_search_k": 10,
    "use_llm_for_column_selection": True
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

# Insert fake data for testing
def populate_fake_data():
    fake_data = [
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

    with engine.connect() as conn:
        conn.execute(participants_table.insert(), fake_data)

# Populate the database with fake data
populate_fake_data()

sql_db = SQLDatabase(engine)
sql_chain = SQLDatabaseChain(llm=OpenAI(), database=sql_db)

# Step 2: Set up the FAISS vector store
# Initialize FAISS index for semantic search
embedding_model = OpenAIEmbeddings()
faiss_index = faiss.IndexFlatL2(embedding_model.embedding_dimension())
faiss_store = FAISS(embedding_model, faiss_index)

# Add unstructured data to FAISS for semantic search
unstructured_data = [
    {"id": 1, "text": "Backend Development, Python, APIs"},
    {"id": 2, "text": "Frontend Development, React, CSS"},
    {"id": 3, "text": "DevOps, CI/CD, Kubernetes"},
    {"id": 4, "text": "Data Science, Machine Learning, R"},
    {"id": 5, "text": "Database Management, SQL, Oracle"},
    {"id": 6, "text": "Mobile Development, Swift, Android"},
    {"id": 7, "text": "Cybersecurity, Ethical Hacking, Firewalls"},
    {"id": 8, "text": "Cloud Computing, AWS, Azure"},
    {"id": 9, "text": "AI Research, NLP, TensorFlow"},
    {"id": 10, "text": "Project Management, Agile, Scrum"}
]

for record in unstructured_data:
    embedding = embedding_model.embed_query(record["text"])
    metadata = {"id": record["id"], "text": record["text"]}
    faiss_store.add_texts([record["text"]], [metadata])

# Step 3: Parse user queries
# Function to process user input and divide into SQL and semantic components
def parse_query(user_query):
    llm = OpenAI()
    prompt = f"""
    Parse the following user query into structured (SQL-like) and unstructured (semantic search) components:

    Query: "{user_query}"

    Respond in JSON format with two keys:
    - sql_conditions: A natural language description of the structured query.
    - semantic_conditions: A list of keywords or phrases for semantic search (e.g., "backend software engineering").
    """

    response = llm(prompt)
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
                "sp_id", "age", "gender", "years_of_experience", "skills", "education_specialization", "past_jobs"
            ]
        }
    }, indent=2)}

    User Request: {sql_description}

    SQL Query:
    """
    
    llm = OpenAI()
    response = llm(prompt)
    sql_query = response.strip()

    with engine.connect() as conn:
        results = conn.execute(sql_query).fetchall()
    return results

# Step 5: Execute Semantic Queries
def execute_semantic_queries(semantic_conditions):
    semantic_results = []
    if CONFIG["use_vector_search"]:
        for condition in semantic_conditions:
            semantic_results.extend(faiss_store.similarity_search(condition, k=CONFIG["vector_search_k"]))
    if CONFIG["use_llm_for_column_selection"]:
        llm = OpenAI()
        prompt = f"""
        Based on the user's query, decide which columns from the database may contain relevant unstructured data.

        Query: {semantic_conditions}
        Available Columns: ["skills", "education_specialization", "past_jobs"]

        Provide a list of column names that may be relevant.
        """
        response = llm(prompt)
        relevant_columns = json.loads(response).get("relevant_columns", [])
        with engine.connect() as conn:
            for column in relevant_columns:
                query = f"SELECT sp_id, {column} FROM participants"
                semantic_results.extend(conn.execute(query).fetchall())
    return semantic_results

# Step 6: Post-process results with LLM
def process_results_with_llm(sql_results, semantic_results, user_query):
    llm = OpenAI()
    prompt = f"""
    Based on the user's query: "{user_query}", analyze and aggregate the following data:

    SQL Results:
    {json.dumps([dict(row) for row in sql_results], default=str)}

    Semantic Results:
    {json.dumps([result['metadata'] if isinstance(result, dict) else dict(result) for result in semantic_results], default=str)}

    Provide a clear and concise response, aggregating the data where necessary (e.g., histograms, summaries).
    """
    
    response = llm(prompt)
    return response

# Step 7: Main Chatbot Functionality
def chatbot(query):
    sql_description, semantic_conditions = parse_query(query)
    sql_results = execute_sql_query(sql_description)
    semantic_results = execute_semantic_queries(semantic_conditions)
    final_response = process_results_with_llm(sql_results, semantic_results, query)
    return final_response

# Main entry point
if __name__ == "__main__":
    print("Chatbot is running. Type your query below (or type 'exit' to quit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        try:
            response = chatbot(user_input)
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

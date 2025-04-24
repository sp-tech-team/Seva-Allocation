import pdb
import os
import pandas as pd
import argparse
from prettytable import PrettyTable
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData, ARRAY, text, inspect
from sqlalchemy.exc import SQLAlchemyError


from pgvector.sqlalchemy import Vector
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from database.participant_data import ParticipantData

from dotenv import load_dotenv

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--database_name',
        type=str,
        help='Name of the database to connect to',
        default="participants_test")
    parser.add_argument(
        '--create_db',
        action='store_true',
        help='Whether to refresh data in the database (default: False)')

    
    return parser.parse_args()

STRUCTURED_TABLE_NAME = "structured_data"
UNSTRUCTURED_TABLE_NAME = "unstructured_data"

class DbConfig:
    def __init__(self, db_user, db_host, db_port, db_name, db_password, openai_api_key):
        self.db_user = db_user
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_password = db_password
        self.openai_api_key = openai_api_key

def pretty_print_sqlalchemy_results(results, max_rows=10):
    """
    Pretty print any SQLAlchemy result set using a markdown-style table.

    Args:
        result: SQLAlchemy result object (e.g., from conn.execute(...)).
        max_rows: Maximum number of rows to print (default = 10).
    """
    pdb.set_trace()
    rows = results.fetchall()
    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows, columns=results.keys())
    print(df.head(max_rows).to_markdown(index=False))

def format_query_result(result, headers=None):
    table = PrettyTable()
    if headers:
        table.field_names = headers  # Set column headers
    for row in result:
        table.add_row(row)  # Add each row of data
    return table.get_string()

class ParticipantDatabasePG:
    def __init__(self, engine):
        self.engine = engine
        self.inspector = inspect(engine)
        self.lc_db = SQLDatabase(engine)
        self.lc_db_query_tool = QuerySQLDataBaseTool(db=self.lc_db)

    def get_structured_table_name(self):
        return STRUCTURED_TABLE_NAME

    def get_unstructured_table_name(self):
        return UNSTRUCTURED_TABLE_NAME

    def get_structured_table_columns_info(self):
        return self.inspector.get_columns(STRUCTURED_TABLE_NAME)
    
    def get_structured_table_column_names(self):
        return [col["name"] for col in self.get_structured_table_columns_info()]

    def get_unstructured_table_columns_info(self):
        return self.inspector.get_columns(UNSTRUCTURED_TABLE_NAME)
    
    def get_unstructured_table_column_names(self):
        return [col["name"] for col in self.get_unstructured_table_columns_info()]

    def list_all_tables_and_columns(self):
        query = """
        SELECT 
            table_name, 
            column_name, 
            data_type
        FROM 
            information_schema.columns
        WHERE 
            table_schema = 'public'
        ORDER BY 
            table_name, ordinal_position;
        """
        with self.engine.connect() as conn:
            results = conn.execute(text(query))
            return results

    def print_all_tables_and_columns(self):
        results = self.list_all_tables_and_columns()
        if results:
            pretty_print_sqlalchemy_results(results, max_rows=100)
        else:
            print("No tables found in the database.")

    def embed_query(self, embedding_model, query: str):
        return embedding_model.embed_query(query)  # Return list[float]

    def run_query(self, query_string):
        try:
            with self.engine.connect() as conn:
                results = conn.execute(text(query_string))
                return {
                    "success": True,
                    "results": results
                }
        except SQLAlchemyError as e:
            return {
                "success": False,
                "error": str(e.__cause__) if e.__cause__ else str(e),
                "query": query_string
            }

    def similarity_search(self, query_vector, table, embedding_columns, text_columns=None, threshold=None, limit=100):
        where_clause = " OR ".join([
            f'"{col}" <=> :query_vector < {threshold}' for col in embedding_columns
        ]) if threshold else "TRUE"

        selected_cols = ', '.join([f'"{col}"' for col in (text_columns or [])])
        sql = f"""
        SELECT 
            "SP ID",
            {selected_cols}
        FROM {table}
        WHERE {where_clause}
        LIMIT {limit};
        """
        pgvector_str = f"[{', '.join(map(str, query_vector))}]"
        with self.engine.connect() as conn:
            results = conn.execute(text(sql), {"query_vector": pgvector_str})
            return results
    
def create_participants_db(db_config: DbConfig):
    # === Set up OpenAI Embeddings ===
    embedding_model = OpenAIEmbeddings()  # uses OPENAI_API_KEY env var

    use_mock_data = True
    if use_mock_data:
        participant_info_df = pd.read_csv('chatbot/test_data/input_participant_info_cleaned_mock2.csv')
    else:
        participant_info_raw_df = pd.read_csv('data/input_participant_info_raw.csv')
        participant_data = ParticipantData(participant_info_raw_df)
        participant_info_df = participant_data.create_participant_info_df()
    structured_cols = ["Gender", "Age", "Total Years of Experience"]
    unstructured_cols = ["Work Experience/Company", "Work Experience/Designation",
                        "Work Experience/Tasks", "Work Experience/Industry",
                        "Education/Qualifications", "Education/Specialization",
                        "Any Additional Skills", "Computer Skills", "Skills", "Languages"]
    structured_df = participant_info_df[["SP ID"] + structured_cols]
    unstructured_df = participant_info_df[["SP ID"] + unstructured_cols]

    # === Prepare structured_df array columns ===
    # structured_df["Languages"] = structured_df["Languages"].apply(lambda x: x if isinstance(x, list) else [])

    # === Create DB connection ===
    engine = create_engine(f"postgresql+psycopg2://{db_config.db_user}@{db_config.db_host}:{db_config.db_port}/{db_config.db_name}")
    metadata = MetaData()

    # === Upload structured_df to PostgreSQL ===
    structured_df.to_sql(
        STRUCTURED_TABLE_NAME,
        engine,
        if_exists="replace",
        index=False,
        dtype={
            "SP ID": String,
            "Gender": String,
            "Age": Integer,
            "Total Years of Experience": Float,
            "Languages": ARRAY(String),
        }
    )

    # === Generate embeddings using LangChain + OpenAI ===
    for col in unstructured_cols:
        texts = unstructured_df[col].fillna("").astype(str).tolist()
        embeddings = embedding_model.embed_documents(texts)
        unstructured_df[f"{col}_embedding"] = embeddings

    # === Define unstructured table with pgvector columns ===
    columns = [Column("SP ID", String)]
    for col in unstructured_cols:
        columns.append(Column(col, String))  # store raw text
        columns.append(Column(f"{col}_embedding", Vector(1536)))  # store embedding

    unstructured_table = Table(UNSTRUCTURED_TABLE_NAME, metadata, *columns)
    metadata.drop_all(engine, [unstructured_table], checkfirst=True)
    metadata.create_all(engine)

    # === Insert records into UNSTRUCTURED_TABLE_NAME ===
    insert_data = []
    for _, row in unstructured_df.iterrows():
        record = {"SP ID": row["SP ID"]}
        for col in unstructured_cols:
            record[col] = row[col]  # original text
            record[f"{col}_embedding"] = row[f"{col}_embedding"]
        insert_data.append(record)

    with engine.begin() as conn:
        conn.execute(unstructured_table.insert(), insert_data)
    
    print("finished")
    return ParticipantDatabasePG(engine)

def load_participant_db(db_config: DbConfig):
    # === Load database ===
    engine = create_engine(f"postgresql+psycopg2://{db_config.db_user}@{db_config.db_host}:{db_config.db_port}/{db_config.db_name}")
    participant_db = ParticipantDatabasePG(engine)
    return participant_db

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    database_name = args.database_name
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    db_config = DbConfig(
        os.getenv("DB_USER"),
        os.getenv("DB_HOST"),
        os.getenv("DB_PORT"),
        database_name,
        os.getenv("DB_PASSWORD"),
        os.getenv("OPENAI_API_KEY")
    )

    # === Test the database functionality ===
    # Load database
    participant_db = None
    if args.create_db:
        participant_db = create_participants_db(db_config)
    else:
        participant_db = load_participant_db(db_config)

     # === Test 1: Print all table columns ===
    print("\n--- All Tables and Columns ---")
    participant_db.print_all_tables_and_columns()

    # === Test 2: Get table column metadata ===
    print("\n--- Structured Table Columns ---")
    print(participant_db.get_structured_table_column_names())

    print("\n--- Unstructured Table Columns ---")
    print(participant_db.get_unstructured_table_column_names())

    # === Test 3: Embed and run similarity search ===
    # === Load vector data to search ===
    embedding_model = OpenAIEmbeddings()
    query = "front end developer"
    vector = participant_db.embed_query(embedding_model, query)

    similarity_results = participant_db.similarity_search(
        query_vector=vector,
        table=UNSTRUCTURED_TABLE_NAME,
        embedding_columns=["Skills_embedding", "Computer Skills_embedding"],
        text_columns=["Skills", "Computer Skills"],
        threshold=0.2,
        limit=3
    )
    print("\n--- Similarity Search Results ---")
    pretty_print_sqlalchemy_results(similarity_results)

     # === Test 4: Run ad hoc SQL query ===
    print("\n--- Custom Query Result ---")
    custom_query = """
    SELECT "SP ID", "Skills"
    FROM unstructured_data
    LIMIT 5;
    """
    custom_result = participant_db.run_query(custom_query)
    if custom_result["success"]:
        print("Query executed successfully.")
        pretty_print_sqlalchemy_results(custom_result)
    else:
        print("Error executing query:", custom_result["error"])
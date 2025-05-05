import pdb
import os
import pandas as pd
import argparse
import json
from prettytable import PrettyTable
from sqlalchemy import create_engine, Table, Column, Integer, Float, String, MetaData, ARRAY, text, inspect
from sqlalchemy.exc import SQLAlchemyError


from pgvector.sqlalchemy import Vector
from langchain_openai import OpenAIEmbeddings
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

from database.participant_data import filter_data_columns

from dotenv import load_dotenv

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--create_db',
        action='store_true',
        help='Whether to refresh data in the database (default: False)')
    parser.add_argument(
        '--input_file_csv',
        type=str,
        help='Input CSV file for creating the database',
        default='data/input_participant_info_cleaned.csv'
    )
    parser.add_argument(
        '--table_base_name',
        type=str,
        help='The base name of the tables group in the database, ex "participants", "participants_mock2"',
        default='participants'
    )
    parser.add_argument(
        '--column_info_config_json',
        type=str,
        help='JSON file for column info configuration',
        default='database/column_info_config.json'
    )
    
    return parser.parse_args()

STRUCTURED_TABLE_NAME_POSTFIX = "structured_data"
UNSTRUCTURED_TABLE_NAME_POSTFIX = "unstructured_data"

class DbConfig:
    def __init__(self, db_user, db_host, db_port, db_name, db_password):
        self.db_user = db_user
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_password = db_password

def pretty_print_sqlalchemy_results(results, max_rows=10):
    """
    Pretty print any SQLAlchemy result set using a markdown-style table.

    Args:
        result: SQLAlchemy result object (e.g., from conn.execute(...)).
        max_rows: Maximum number of rows to print (default = 10).
    """
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
    def __init__(self, engine, table_base_name):
        self.engine = engine
        self.table_base_name = table_base_name
        self.inspector = inspect(engine)
        self.lc_db = SQLDatabase(engine)
        self.lc_db_query_tool = QuerySQLDataBaseTool(db=self.lc_db)

    def reset_table_base_name(self, table_base_name):
        self.table_base_name = table_base_name

    def get_table_names_limited(self):
        return [self.get_structured_table_name(), self.get_unstructured_table_name()]
    def get_structured_table_name(self):
        return self.table_base_name + '_' + STRUCTURED_TABLE_NAME_POSTFIX

    def get_unstructured_table_name(self):
        return self.table_base_name + '_' + UNSTRUCTURED_TABLE_NAME_POSTFIX

    def get_structured_table_columns_info(self):
        return self.inspector.get_columns(self.get_structured_table_name())
    
    def get_structured_table_column_names(self):
        return [col["name"] for col in self.get_structured_table_columns_info()]

    def get_unstructured_table_columns_info(self):
        return self.inspector.get_columns(self.get_unstructured_table_name())
    
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
        #return {"result": self.participant_db.lc_db_query_tool.invoke(sql_query)}
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

    def similarity_search(self, query_vector, embedding_columns, text_columns=None, threshold=None, limit=100):
        where_clause = " OR ".join([
            f'"{col}" <=> :query_vector < {threshold}' for col in embedding_columns
        ]) if threshold else "TRUE"

        selected_cols = ', '.join([f'"{col}"' for col in (text_columns or [])])
        sql = f"""
        SELECT 
            "SP ID",
            {selected_cols}
        FROM {self.get_unstructured_table_name()}
        WHERE {where_clause}
        LIMIT {limit};
        """
        pgvector_str = f"[{', '.join(map(str, query_vector))}]"
        with self.engine.connect() as conn:
            results = conn.execute(text(sql), {"query_vector": pgvector_str})
            return results

def make_unstructured_table(engine, unstructured_table_name, unstructured_df):
    unstructured_cols = [col for col in unstructured_df.columns.tolist() if col != "SP ID"]
    # === Generate embeddings using LangChain + OpenAI ===
    embedding_model = OpenAIEmbeddings()
    for col in unstructured_cols:
        texts = unstructured_df[col].fillna("").astype(str).tolist()
        embeddings = embedding_model.embed_documents(texts)
        unstructured_df[f"{col}_embedding"] = embeddings

    # === Define unstructured table with pgvector columns ===
    columns = [Column("SP ID", String)]
    for col in unstructured_cols:
        columns.append(Column(col, String))  # store raw text
        columns.append(Column(f"{col}_embedding", Vector(1536)))  # store embedding
    
    metadata = MetaData()
    unstructured_table = Table(unstructured_table_name, metadata, *columns)
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

def create_participants_db(participant_info_df: pd.DataFrame, column_info_config: dict, db_config: DbConfig, table_base_name: str = ''):
    column_info_config["data_columns"]
    structured_cols = filter_data_columns(column_info_config["data_columns"], data_mode="structured", upload_db=True)
    unstructured_cols = filter_data_columns(column_info_config["data_columns"], data_mode="unstructured", upload_db=True)
    structured_df = participant_info_df[[column_info_config["data_key_column"]] + structured_cols]
    unstructured_df = participant_info_df[[column_info_config["data_key_column"]] + unstructured_cols]

    # === Create DB connection ===
    connection_str = f"postgresql+psycopg2://{db_config.db_user}:{db_config.db_password}@{db_config.db_host}:{db_config.db_port}/{db_config.db_name}"
    engine = create_engine(connection_str)

    # === Upload structured_df to PostgreSQL ===
    structured_table_name = table_base_name + '_' + STRUCTURED_TABLE_NAME_POSTFIX
    structured_df.to_sql(
        structured_table_name,
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

    # === Create UNstructured table with pgvector columns ===
    unstructured_table_name = table_base_name + '_' + UNSTRUCTURED_TABLE_NAME_POSTFIX
    make_unstructured_table(engine, unstructured_table_name, unstructured_df)
    
    print("finished")
    return ParticipantDatabasePG(engine, table_base_name)

def load_participant_db(db_config: DbConfig, table_base_name: str = ''):
    # === Load database ===
    connection_str = f"postgresql+psycopg2://{db_config.db_user}:{db_config.db_password}@{db_config.db_host}:{db_config.db_port}/{db_config.db_name}"
    engine = create_engine(connection_str)
    participant_db = ParticipantDatabasePG(engine, table_base_name)
    return participant_db

if __name__ == "__main__":
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    args = parse_args()
    # Load database
    db_config = DbConfig(
        os.getenv("SUPABASE_USER"),
        os.getenv("SUPABASE_HOST"),
        os.getenv("SUPABASE_PORT"),
        os.getenv("SUPABASE_NAME"),
        os.getenv("SUPABASE_PASSWORD")
    )
    participant_db = None
    if args.create_db:
        with open(args.column_info_config_json, 'r') as f:
            column_info_config = json.load(f)
        participant_info_df = pd.read_csv(args.input_file_csv)
        participant_db = create_participants_db(participant_info_df, column_info_config, db_config, args.table_base_name)
    else:
        participant_db = load_participant_db(db_config, args.table_base_name)

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
        embedding_columns=["Skills_embedding", "Computer Skills_embedding"],
        text_columns=["Skills", "Computer Skills"],
        threshold=0.2,
        limit=3
    )
    print("\n--- Similarity Search Results ---")
    pretty_print_sqlalchemy_results(similarity_results)

     # === Test 4: Run ad hoc SQL query ===
    print("\n--- Custom Query Result ---")
    custom_query = f"""
    SELECT "SP ID", "Skills"
    FROM {participant_db.get_unstructured_table_name()}
    LIMIT 5;
    """
    custom_result = participant_db.run_query(custom_query)
    if custom_result["success"]:
        print("Query executed successfully.")
        pretty_print_sqlalchemy_results(custom_result['results'])
    else:
        print("Error executing query:", custom_result["error"])
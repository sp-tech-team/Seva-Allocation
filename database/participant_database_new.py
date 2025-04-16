import pandas as pd
from sqlalchemy import create_engine, inspect
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
# Assuming you have an embedding model class available, e.g.:
# from langchain_openai import OpenAIEmbeddings # Or any other embedding model
from pydantic import BaseModel, Field
from typing import Optional, List
import ast
from prettytable import PrettyTable
from langchain_openai import OpenAIEmbeddings
from preprocessing.participant_data import ParticipantData
from dotenv import load_dotenv
load_dotenv()



# --- Helper Functions (Unchanged) ---

def format_query_result(result, headers=None):
    """Formats SQL query results using PrettyTable."""
    table = PrettyTable()
    if headers:
        table.field_names = headers  # Set column headers
    # Ensure result is iterable (it might be a string representation)
    try:
        # Attempt to parse if it looks like a list literal string
        if isinstance(result, str) and result.startswith('[') and result.endswith(']'):
            parsed_result = ast.literal_eval(result)
        else:
            # Assume it's already an iterable or handle other cases if necessary
            parsed_result = result if hasattr(result, '__iter__') else [] # Basic check
    except (ValueError, SyntaxError):
        # Handle cases where result is not a valid literal or not iterable
        print(f"Warning: Could not parse result for PrettyTable: {result}")
        parsed_result = [] # Default to empty list on error

    for row in parsed_result:
        table.add_row(row)  # Add each row of data
    return table.get_string()

def get_column_names_sql_query(execute_query_tool, table_name):
    """Gets column names for a given table using PRAGMA query."""
    query = f"PRAGMA table_info({table_name});"
    result = execute_query_tool.invoke(query)
    try:
        parsed_result = ast.literal_eval(result)
        # Extract column names (second element in each tuple)
        column_names = [col[1] for col in parsed_result]
        return column_names
    except (ValueError, SyntaxError, IndexError) as e:
        print(f"Error parsing PRAGMA result for {table_name}: {e}")
        print(f"Raw result: {result}")
        return [] # Return empty list on error

def explode_columns(df, paired_cols, independent_cols):
    """Explodes DataFrame columns."""
    # Ensure paired columns exist before exploding
    valid_paired_cols = [col for col in paired_cols if col in df.columns]
    if valid_paired_cols:
         df = df.explode(valid_paired_cols, ignore_index=True)

    for col in independent_cols:
         if col in df.columns: # Check if independent column exists
             # Ensure the column contains list-like objects before exploding
             if df[col].apply(lambda x: isinstance(x, (list, tuple))).any():
                 df = df.explode(col, ignore_index=True) # ignore_index=True for newer pandas versions
             # else: column might not need exploding or isn't list-like

    df = df.fillna('NA')
    return df


# --- Pydantic Model (Unchanged) ---
class PydanticUnstructuredCategories(BaseModel):
    """
    A user profile model with optional fields for unstructured data categories.
    """
    work_experience_company: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of companies the user has worked for."
    )
    work_experience_designation: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of job designations held by the user."
    )
    work_experience_tasks: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of tasks the user performed in past jobs."
    )
    work_experience_industry: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of industries the user has worked in."
    )
    education_qualifications: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of the user's educational qualifications."
    )
    education_specialization: Optional[List[str]] = Field(
        default_factory=list,
        description="The user's educational specializations."
    )
    any_additional_skills: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of additional skills the user possesses."
    )
    computer_skills: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of computer-related skills the user has."
    )
    skills: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of general skills the user has."
    )
    languages: Optional[List[str]] = Field(
        default_factory=list,
        description="A list of languages spoken by the user."
    )

# --- Modified ParticipantDatabase Class ---
class ParticipantDatabase:
    """
    Manages participant data stored in structured and unstructured tables
    within a single SQLite database.
    """
    def __init__(self, engine, sql_db, execute_query_tool,
                 pydantic_unstructured_categories,
                 structured_table_name="participants_structured", # Store table names
                 unstructured_table_name="participants_unstructured"):
        self.engine = engine # Single engine
        self.sql_db = sql_db # Single SQLDatabase object
        self.execute_query_tool = execute_query_tool # Single query tool
        self.structured_table_name = structured_table_name
        self.unstructured_table_name = unstructured_table_name
        self.pydantic_unstructured_categories = pydantic_unstructured_categories

    def get_structured_table_name(self):
        """Returns the name of the structured data table."""
        # Option 1: Return the stored name (simpler)
        return self.structured_table_name

    def get_unstructured_table_name(self):
        """Returns the name of the unstructured data table."""
        # Option 1: Return the stored name (simpler)
        return self.unstructured_table_name

    def get_structured_column_names(self):
        """Gets column names for the structured table."""
        table_name = self.get_structured_table_name()
        if table_name:
            # Use the single execute_query_tool
            return get_column_names_sql_query(self.execute_query_tool, table_name)
        return []

    def get_unstructured_column_names(self):
        """Gets column names for the unstructured table."""
        table_name = self.get_unstructured_table_name()
        if table_name:
            # Use the single execute_query_tool
            return get_column_names_sql_query(self.execute_query_tool, table_name)
        return []

    def get_unstructured_data_dicts(self):
        """Retrieves all data from the unstructured table as a list of dictionaries."""
        table_name = self.get_unstructured_table_name()
        if not table_name:
            return []
        query = f"SELECT * FROM {table_name};"
        # Use the single execute_query_tool
        result = self.execute_query_tool.invoke(query)
        try:
            unstructured_data = ast.literal_eval(result)
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse query result for unstructured data: {result}")
            return []

        unstructured_cols = self.get_unstructured_column_names()
        if not unstructured_cols: # Check if column names were retrieved
             print(f"Warning: Could not get column names for table {table_name}")
             return [] # Cannot create dicts without column names

        unstructured_data_dicts = [dict(zip(unstructured_cols, row)) for row in unstructured_data]
        return unstructured_data_dicts

    def make_faiss_index(self, embedding_model):
        """Creates a FAISS index from the unstructured data."""
        # Get embedding dimension (handle potential errors)
        try:
            sample_embedding = embedding_model.embed_query("sample text")
            embedding_dim = len(sample_embedding)
        except Exception as e:
            print(f"Error getting embedding dimension: {e}")
            return None # Cannot create index without dimension

        faiss_index = faiss.IndexFlatL2(embedding_dim)
        faiss_store = FAISS(
            embedding_function=embedding_model,
            index=faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        unstructured_data_dicts = self.get_unstructured_data_dicts()
        if not unstructured_data_dicts: # Check if data was retrieved
            print("Warning: No unstructured data found to build FAISS index.")
            return faiss_store # Return empty store

        batch_unstructured_texts = []
        batch_unstructured_metadata = []
        for record in unstructured_data_dicts:
            # Ensure 'SP ID' exists, provide a default if not
            sp_id = record.get("SP ID", "UNKNOWN_ID")

            combined_text = " ".join([f"{key}: {str(value)}\n"
                                for key, value in record.items() if key != "SP ID"])
            metadata = {"id": sp_id, "text": combined_text, "record": record}
            batch_unstructured_texts.append(combined_text)
            batch_unstructured_metadata.append(metadata)

        if batch_unstructured_texts: # Check if there are texts to add
            faiss_store.add_texts(batch_unstructured_texts, batch_unstructured_metadata)
        return faiss_store

# --- Modified Database Creation Function ---
def create_participant_database(
    input_csv_file='data/input_participant_info_raw.csv',
    # Single database file path
    db_file="sqlite:///chatbot/data/participants_combined.db",
    structured_table_name = "participants_structured", # Define table names
    unstructured_table_name = "participants_unstructured"
):
    """
    Creates a single SQLite database with structured and unstructured participant tables.
    """
    # Load data (Assuming ParticipantData class exists and works)
    try:
        participant_info_raw_df = pd.read_csv(input_csv_file)
        # Replace with your actual data loading/preprocessing logic if ParticipantData is different
        # For demonstration, assuming participant_info_raw_df is ready
        participant_data = ParticipantData(participant_info_raw_df) # If you have this class
        participant_info_df = participant_data.create_participant_info_df() # If you have this class
        participant_info_df = participant_info_raw_df # Placeholder if ParticipantData class is not provided
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at {input_csv_file}")
        return None
    except Exception as e:
        print(f"Error loading or processing CSV: {e}")
        return None

    # Define columns for each table (Adjust based on your actual CSV)
    # Make sure these column names EXACTLY match your CSV headers
    structured_cols = ["SP ID", "Gender", "Age", "Work Experience/From Date", "Work Experience/To Date", "Languages"]
    unstructured_cols = ["SP ID", "Work Experience/Company", "Work Experience/Designation",
                         "Work Experience/Tasks", "Work Experience/Industry",
                         "Education/Qualifications", "Education/Specialization",
                         "Any Additional Skills", "Computer Skills", "Skills", "Languages"] # Languages can be in both

    # Ensure all required columns exist in the DataFrame
    missing_structured = [col for col in structured_cols if col not in participant_info_df.columns]
    missing_unstructured = [col for col in unstructured_cols if col not in participant_info_df.columns]

    if missing_structured:
        print(f"Warning: Missing structured columns in CSV: {missing_structured}")
        # Decide how to handle: error out, or proceed with available columns?
        # Let's proceed with available columns for now:
        structured_cols = [col for col in structured_cols if col in participant_info_df.columns]
        if not structured_cols:
             print("Error: No structured columns found to create the table.")
             return None


    if missing_unstructured:
        print(f"Warning: Missing unstructured columns in CSV: {missing_unstructured}")
        # Proceed with available columns:
        unstructured_cols = [col for col in unstructured_cols if col in participant_info_df.columns]
        if not unstructured_cols:
            print("Error: No unstructured columns found to create the table.")
            return None

    # Check if 'SP ID' is present as it's crucial
    if "SP ID" not in participant_info_df.columns:
        print("Error: 'SP ID' column is missing, which is required for linking tables.")
        return None

    # Split data into two separate DataFrames using available columns
    df_structured = participant_info_df[structured_cols].copy() # Use .copy() to avoid SettingWithCopyWarning
    df_unstructured = participant_info_df[unstructured_cols].copy()

    # --- Preprocess Structured Data ---
    # Define columns for exploding - check if they exist first
    paired_columns = [col for col in ['Work Experience/From Date', 'Work Experience/To Date'] if col in df_structured.columns]
    independent_columns = [col for col in ['Languages'] if col in df_structured.columns] # Languages might be list-like

    # Convert potential list-like strings in 'Languages' to actual lists for explode
    # This is a common issue when reading from CSV
    if 'Languages' in df_structured.columns:
         # Example conversion: assumes comma-separated strings if not already lists
         df_structured['Languages'] = df_structured['Languages'].apply(
             lambda x: x.split(',') if isinstance(x, str) else x
         )

    df_structured = explode_columns(df_structured, paired_columns, independent_columns)


    # --- Preprocess Unstructured Data ---
    # Convert lists to comma-separated strings for storage in SQLite
    # (SQLite doesn't natively support list types well)
    for col in df_unstructured.columns:
        # Apply only to columns that might contain lists
        if df_unstructured[col].apply(lambda x: isinstance(x, list)).any():
             df_unstructured[col] = df_unstructured[col].apply(
                 lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x
             )
    # Fill NA values if any were introduced
    df_unstructured = df_unstructured.fillna('NA')


    # --- Database Operations ---
    # Create a single engine for the combined database
    engine = create_engine(db_file)

    # Write both DataFrames as tables to the *same* database engine
    try:
        df_structured.to_sql(structured_table_name, engine, if_exists="replace", index=False)
        print(f"Table '{structured_table_name}' created successfully.")
        df_unstructured.to_sql(unstructured_table_name, engine, if_exists="replace", index=False)
        print(f"Table '{unstructured_table_name}' created successfully.")
    except Exception as e:
        print(f"Error writing tables to database: {e}")
        return None

    # Create a single SQLDatabase object
    sql_db = SQLDatabase(engine)

    # Create a single query execution tool
    execute_query_tool = QuerySQLDataBaseTool(db=sql_db)

    # Instantiate ParticipantDatabase with the single set of resources
    participant_db = ParticipantDatabase(
        engine=engine,
        sql_db=sql_db,
        execute_query_tool=execute_query_tool,
        pydantic_unstructured_categories=PydanticUnstructuredCategories, # Pass the class itself
        structured_table_name=structured_table_name,
        unstructured_table_name=unstructured_table_name
    )

    print(f"ParticipantDatabase initialized with single database: {db_file}")
    return participant_db

# --- Example Usage (Requires an embedding model) ---
if __name__ == "__main__":
    
    embedding_model = OpenAIEmbeddings()
    # Create the database and get the handler object
    # Ensure 'data/input_participant_info_raw.csv' exists or change the path
    db_handler = create_participant_database(input_csv_file='data/input_participant_info_raw.csv')

    if db_handler:
        print("\n--- Database Info ---")
        print(f"Structured Table: {db_handler.get_structured_table_name()}")
        structured_cols = db_handler.get_structured_column_names()
        print(f"Structured Columns: {structured_cols}")

        print(f"Unstructured Table: {db_handler.get_unstructured_table_name()}")
        unstructured_cols = db_handler.get_unstructured_column_names()
        print(f"Unstructured Columns: {unstructured_cols}")

        # Example: Querying data (using the tool directly for demo)
        print("\n--- Sample Structured Data (First 5) ---")
        query_structured = f"SELECT * FROM {db_handler.get_structured_table_name()} LIMIT 100;"
        result_structured = db_handler.execute_query_tool.invoke(query_structured)
        print(format_query_result(result_structured, headers=structured_cols))


        print("\n--- Sample Unstructured Data (First 5) ---")
        query_unstructured = f"SELECT * FROM {db_handler.get_unstructured_table_name()} LIMIT 1;"
        result_unstructured = db_handler.execute_query_tool.invoke(query_unstructured)
        print(format_query_result(result_unstructured, headers=unstructured_cols))


        # Example: Get unstructured data as dicts
        # unstructured_dicts = db_handler.get_unstructured_data_dicts()
        # if unstructured_dicts:
        #     print(f"\n--- First Unstructured Record as Dict ---")
        #     print(unstructured_dicts[0])

        # Example: Create FAISS index
        # print("\n--- Creating FAISS Index ---")
        # faiss_vector_store = db_handler.make_faiss_index(embedding_model)
        # if faiss_vector_store:
        #      # You can now use the faiss_vector_store for similarity searches
        #      print(f"FAISS index created. Number of documents: {faiss_vector_store.index.ntotal}")
        #      # Example search (will only work well with real embeddings)
        #      # search_results = faiss_vector_store.similarity_search("looking for java skills", k=2)
        #      # print("\n--- Sample FAISS Search Results ---")
        #      # for doc in search_results:
        #      #    print(f"ID: {doc.metadata.get('id', 'N/A')}, Score: N/A (IndexFlatL2 doesn't store scores directly), Text: {doc.page_content[:100]}...") # Score not directly available this way
        # else:
        #      print("Failed to create FAISS index.")

    else:
        print("Database creation failed.")
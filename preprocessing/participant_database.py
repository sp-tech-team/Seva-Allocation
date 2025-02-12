from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

from sqlalchemy import create_engine, inspect, Table, Column, Integer, String, MetaData
from typing import List, Optional
from pydantic import BaseModel, Field

import pandas as pd
import ast
from prettytable import PrettyTable
import pdb

from preprocessing.participant_data import ParticipantData

def format_query_result(result, headers=None):
    table = PrettyTable()
    if headers:
        table.field_names = headers  # Set column headers
    for row in result:
        table.add_row(row)  # Add each row of data
    return table.get_string()

def get_column_names_sql_query(execute_query_tool, table_name):
    query = f"PRAGMA table_info({table_name});"
    result = execute_query_tool.invoke(query)
    parsed_result = ast.literal_eval(result)
    # Extract column names (second element in each tuple)
    column_names = [col[1] for col in parsed_result]
    return column_names

class ParticipantDatabase:
    def __init__(self, engine_structured, engine_unstructured,
                 sql_db_structured, sql_db_unstructured,
                 execute_structured_query_tool, execute_unstructured_query_tool,
                 pydantic_unstructured_categories):
        self.engine_structured = engine_structured
        self.engine_unstructured = engine_unstructured
        self.sql_db_structured = sql_db_structured
        self.sql_db_unstructured = sql_db_unstructured
        self.execute_structured_query_tool = execute_structured_query_tool
        self.execute_unstructured_query_tool = execute_unstructured_query_tool
        self.pydantic_unstructured_categories = pydantic_unstructured_categories
    
    def get_structured_table_name(self):
        inspector = inspect(self.engine_structured)
        table_names = inspector.get_table_names()
        return table_names[0]

    def get_unstructured_table_name(self):
        inspector = inspect(self.engine_unstructured)
        table_names = inspector.get_table_names()
        return table_names[0]
    
    def get_structured_column_names(self):
        table_name = self.get_structured_table_name()
        return get_column_names_sql_query(self.execute_structured_query_tool, table_name)
    
    def get_unstructured_column_names(self):
        table_name = self.get_unstructured_table_name()
        return get_column_names_sql_query(self.execute_unstructured_query_tool, table_name)

    def get_unstructured_data_dicts(self):
        table_name = self.get_unstructured_table_name()
        query = f"SELECT * FROM {table_name};"
        result = self.execute_unstructured_query_tool.invoke(query)
        unstructured_data = ast.literal_eval(result)
        unstructured_cols = self.get_unstructured_column_names()
        unstructured_data_dicts = [dict(zip(unstructured_cols, row)) for row in unstructured_data]
        return unstructured_data_dicts
    
    def make_faiss_index(self, embedding_model):
        faiss_index = faiss.IndexFlatL2(len(embedding_model.embed_query("sample text")))
        faiss_store = FAISS(
            embedding_function=embedding_model,
            index=faiss_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        unstructured_data_dicts = self.get_unstructured_data_dicts()
        batch_unstructured_texts = []
        batch_unstructured_metadata = []
        for record in unstructured_data_dicts:
            combined_text = " ".join([f"{key}: {str(value)}\n" 
                                for key, value in record.items() if key != "SP ID"])
            metadata = {"id": record["SP ID"], "text": combined_text, "record": record}
            batch_unstructured_texts.append(combined_text)
            batch_unstructured_metadata.append(metadata)
        faiss_store.add_texts(batch_unstructured_texts, batch_unstructured_metadata)
        return faiss_store


class PydanticUnstructuredCategories(BaseModel):
    """
    A user profile model with optional fields.
    If any field is omitted, it defaults to an empty list.
    However, passing 'null' (None) explicitly is still allowed.
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

def explode_columns(df, paired_cols, independent_cols):
    df = df.explode(paired_cols, ignore_index=True)
    for col in independent_cols:
        df[col] = df[col].explode().reset_index(drop=True)
    df = df.fillna('NA')
    return df

def create_participant_database(structured_db_file = "sqlite:///chatbot/data/participants_structured.db",
                                unstructured_db_file = "sqlite:///chatbot/data/participants_unstructured.db"):
    # Load data
    participant_info_raw_df = pd.read_csv('data/input_participant_info_raw.csv')
    participant_data = ParticipantData(participant_info_raw_df)
    participant_info_df = participant_data.create_participant_info_df()

    structured_cols = ["SP ID", "Gender", "Age", "Work Experience/From Date", "Work Experience/To Date", "Languages"]
    unstructured_cols = ["SP ID", "Work Experience/Company", "Work Experience/Designation",
                        "Work Experience/Tasks", "Work Experience/Industry",
                        "Education/Qualifications", "Education/Specialization",
                        "Any Additional Skills", "Computer Skills", "Skills", "Languages"]

    # Split data into two separate DataFrames
    df_structured = participant_info_df[structured_cols]
    paired_columns = ['Work Experience/From Date', 'Work Experience/To Date']
    independent_columns = ['Languages']
    df_structured = explode_columns(df_structured, paired_columns, independent_columns)

    df_unstructured = participant_info_df[unstructured_cols]
    df_unstructured = df_unstructured.map(lambda x: ", ".join(x) if isinstance(x, list) else x)

    engine_structured = create_engine(structured_db_file)
    engine_unstructured = create_engine(unstructured_db_file)

    df_structured.to_sql("participants_structured", engine_structured, if_exists="replace", index=False)
    df_unstructured.to_sql("participants_unstructured", engine_unstructured, if_exists="replace", index=False)
    sql_db_structured = SQLDatabase(engine_structured)
    sql_db_unstructured = SQLDatabase(engine_unstructured)
    execute_structured_query_tool = QuerySQLDatabaseTool(db=sql_db_structured)
    execute_unstructured_query_tool = QuerySQLDatabaseTool(db=sql_db_unstructured)
    return ParticipantDatabase(engine_structured, engine_unstructured,
                               sql_db_structured, sql_db_unstructured,
                               execute_structured_query_tool, execute_unstructured_query_tool,
                               PydanticUnstructuredCategories)

class MockPydanticUnstructuredCategories(BaseModel):
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

def create_mock_participant_database(structured_db_file = 'sqlite:///chatbot/data/participants_structured_mock.db',
                                     unstructured_db_file = 'sqlite:///chatbot/data/participants_unstructured_mock.db'):
    engine_structured = create_engine(structured_db_file)
    engine_unstructured = create_engine(unstructured_db_file)
    structured_metadata = MetaData()
    unstructured_metadata = MetaData()

    participants_structured_table = Table(
        'participants_structured', structured_metadata,
        Column('sp_id', Integer, primary_key=True),
        Column('age', Integer),
        Column('gender', String),
        Column('years_of_experience', Integer)
    )
    participants_unstructured_table = Table(
        'participants_unstructured', unstructured_metadata,
        Column('sp_id', Integer, primary_key=True),
        Column('skills', String),
        Column('education_specialization', String),
        Column('past_jobs', String)
    )

    structured_metadata.create_all(engine_structured)
    unstructured_metadata.create_all(engine_unstructured)

    structured_data = [
        {"SP ID": 1, "age": 35, "gender": "Male", "years_of_experience": 10},
        {"SP ID": 2, "age": 28, "gender": "Female", "years_of_experience": 5},
        {"SP ID": 3, "age": 40, "gender": "Male", "years_of_experience": 15},
        {"SP ID": 4, "age": 32, "gender": "Female", "years_of_experience": 8},
        {"SP ID": 5, "age": 45, "gender": "Male", "years_of_experience": 20},
        {"SP ID": 6, "age": 29, "gender": "Female", "years_of_experience": 6},
        {"SP ID": 7, "age": 38, "gender": "Male", "years_of_experience": 12},
        {"SP ID": 8, "age": 27, "gender": "Female", "years_of_experience": 4},
        {"SP ID": 9, "age": 36, "gender": "Male", "years_of_experience": 14},
        {"SP ID": 10, "age": 33, "gender": "Female", "years_of_experience": 9}
    ]
    unstructured_data = [
        {"SP ID": 1, "skills": "Backend Development, Python, APIs", "education_specialization": "Computer Science", "past_jobs": "Software Engineer at XYZ"},
        {"SP ID": 2, "skills": "Frontend Development, React, CSS", "education_specialization": "Information Technology", "past_jobs": "UI Developer at ABC"},
        {"SP ID": 3, "skills": "DevOps, CI/CD, Kubernetes", "education_specialization": "Systems Engineering", "past_jobs": "DevOps Engineer at DEF"},
        {"SP ID": 4, "skills": "Data Science, Machine Learning, R", "education_specialization": "Data Science", "past_jobs": "Data Scientist at GHI"},
        {"SP ID": 5, "skills": "Database Management, SQL, Oracle", "education_specialization": "Database Administration", "past_jobs": "DBA at JKL"},
        {"SP ID": 6, "skills": "Mobile Development, Swift, Android", "education_specialization": "Mobile Computing", "past_jobs": "Mobile Developer at MNO"},
        {"SP ID": 7, "skills": "Cybersecurity, Ethical Hacking, Firewalls", "education_specialization": "Cybersecurity", "past_jobs": "Security Analyst at PQR"},
        {"SP ID": 8, "skills": "Cloud Computing, AWS, Azure", "education_specialization": "Cloud Technology", "past_jobs": "Cloud Engineer at STU"},
        {"SP ID": 9, "skills": "AI Research, NLP, TensorFlow", "education_specialization": "Artificial Intelligence", "past_jobs": "AI Specialist at VWX"},
        {"SP ID": 10, "skills": "Project Management, Agile, Scrum", "education_specialization": "Business Administration", "past_jobs": "Project Manager at YZA"}
    ]

    with engine_structured.begin() as conn:
        conn.execute(participants_structured_table.delete())
        conn.execute(participants_structured_table.insert(), structured_data)

    with engine_unstructured.begin() as conn:
        conn.execute(participants_unstructured_table.delete())
        conn.execute(participants_unstructured_table.insert(), unstructured_data)

    sql_db_structured = SQLDatabase(engine_structured)
    sql_db_unstructured = SQLDatabase(engine_unstructured)
    execute_structured_query_tool = QuerySQLDatabaseTool(db=sql_db_structured)
    execute_unstructured_query_tool = QuerySQLDatabaseTool(db=sql_db_unstructured)

    return ParticipantDatabase(engine_structured, engine_unstructured,
                                   sql_db_structured, sql_db_unstructured,
                                   execute_structured_query_tool, execute_unstructured_query_tool,
                                   MockPydanticUnstructuredCategories)

if __name__ == '__main__':
    participant_data = create_participant_database()
    # participant_data = create_mock_participant_database()
    query = \
    """SELECT "SP ID", "Work Experience/To Date", "Work Experience/From Date"
    FROM participants_structured
    LIMIT 10000;"""
    result = participant_data.execute_structured_query_tool.invoke(query)
    if result:
        print(format_query_result(ast.literal_eval(result)))
    else:
        print("No results found.")
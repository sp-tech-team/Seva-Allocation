from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

from sqlalchemy import create_engine, inspect, Table, Column, Integer, String, MetaData
from typing import List, Optional
from pydantic import BaseModel, Field

import pandas as pd
import ast
from prettytable import PrettyTable
import pdb

from preprocessing.participant_data import create_participant_db_df

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

def create_participant_data():
    # Load data
    target_columns = ['SP ID', 'Work Experience/Company', 'Work Experience/Designation',
       'Work Experience/Tasks', 'Work Experience/Industry',
       'Education/Qualifications', 'Education/Specialization',
       'Any Additional Skills', 'Computer Skills', 'Skills',
       'Languages', 'Gender', 'Age', 'Work Experience/From Date', 'Work Experience/To Date']
    participant_info_raw_df = pd.read_csv('../data/input_participant_info_raw.csv')
    participant_df = create_participant_db_df(participant_info_raw_df, target_columns)

    # count_mismatch = participant_df.apply(lambda row: {col: len(row[col]) if isinstance(row[col], list) else 1 for col in participant_df.columns}, axis=1)
    # mismatch_df = pd.DataFrame(count_mismatch.tolist())
    # paired_columns = ['Work Experience/From Date', 'Work Experience/To Date']
    # mismatch_df['Paired Mismatch'] = mismatch_df[paired_columns[0]] != mismatch_df[paired_columns[1]]

    structured_cols = ["SP ID", "Gender", "Age", "Work Experience/From Date", "Work Experience/To Date", "Languages"]
    unstructured_cols = ["SP ID", "Work Experience/Company", "Work Experience/Designation",
                        "Work Experience/Tasks", "Work Experience/Industry",
                        "Education/Qualifications", "Education/Specialization",
                        "Any Additional Skills", "Computer Skills", "Skills", "Languages"]

    # Split data into two separate DataFrames
    df_structured = participant_df[structured_cols]
    paired_columns = ['Work Experience/From Date', 'Work Experience/To Date']
    independent_columns = ['Languages']
    df_structured = explode_columns(df_structured, paired_columns, independent_columns)

    df_unstructured = participant_df[unstructured_cols]
    df_unstructured = df_unstructured.map(lambda x: ", ".join(x) if isinstance(x, list) else x)

    engine_structured = create_engine("sqlite:///participants_structured.db")
    engine_unstructured = create_engine("sqlite:///participants_unstructured.db")

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

def create_mock_participant_data():
    engine_structured = create_engine('sqlite:///data/participants_structured_mock.db')
    engine_unstructured = create_engine('sqlite:///data/participants_unstructured_mock.db')
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
    participant_data = create_participant_data()
    # participant_data = create_mock_participant_data()

    # column_names = get_column_names_sql_query(participant_data.execute_structured_query_tool, "participants_structured")
    # print(column_names)

    # structured_table_name = participant_data.get_structured_table_name()
    # print("structured table name: ", structured_table_name)
    # unstructured_table_name = participant_data.get_unstructured_table_name()
    # print("unstructured table name: ", unstructured_table_name)
    # structured_col_names = participant_data.get_structured_column_names()
    # print("structured col names: ", structured_col_names)
    # unstructured_col_names = participant_data.get_unstructured_column_names()
    # print("unstructured col names: ", unstructured_col_names)

    # unstructured_data_dicts = participant_data.get_unstructured_data_dicts()
    # print("unstructured data dicts:")
    # print(unstructured_data_dicts)

#     query = \
#     """SELECT "SP ID", "Gender", "Age", "Work Experience/From Date", "Work Experience/To Date"
# FROM participants_structured
# WHERE "Work Experience/To Date" IS NOT NULL AND julianday("Work Experience/From Date") IS NOT NULL
#   AND (julianday("Work Experience/To Date") - julianday("Work Experience/From Date")) / 365 > 10
# ORDER BY "Age" DESC
# LIMIT 10000;"""

#     query = \
#     """SELECT "SP ID", julianday("Work Experience/To Date"), julianday("Work Experience/From Date")
# FROM participants_structured
# WHERE "Work Experience/To Date" IS NOT NULL
# LIMIT 10;"""
    # query = \
    # """SELECT "SP ID", "Gender", "Age", (julianday("Work Experience/To Date") - julianday("Work Experience/From Date")) / 365
    # FROM participants_structured
    # WHERE "Work Experience/From Date" IS NOT 'NA' AND "Work Experience/To Date" IS NOT 'NA' AND (julianday("Work Experience/To Date") - julianday("Work Experience/From Date")) / 365 > 5
    # ORDER BY "Age" DESC
    # LIMIT 10000;"""
    query = \
    """SELECT "SP ID", "Work Experience/To Date", "Work Experience/From Date"
    FROM participants_structured
    LIMIT 10000;"""
    result = participant_data.execute_structured_query_tool.invoke(query)
    if result:
        print(format_query_result(ast.literal_eval(result)))
    else:
        print("No results found.")
# 
# julianday("Work Experience/To Date"), julianday("Work Experience/From Date")
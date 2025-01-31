from langchain_community.utilities.sql_database import SQLDatabase

from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData


def create_fake_participant_data():
    structured_engine = create_engine('sqlite:///participants_structured.db')
    unstructured_engine = create_engine('sqlite:///participants_unstructured.db')
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

    structured_metadata.create_all(structured_engine)
    unstructured_metadata.create_all(unstructured_engine)

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

    with structured_engine.begin() as conn:
        conn.execute(participants_structured_table.delete())
        conn.execute(participants_structured_table.insert(), structured_data)

    with unstructured_engine.begin() as conn:
        conn.execute(participants_unstructured_table.delete())
        conn.execute(participants_unstructured_table.insert(), unstructured_data)

    sql_db_structured = SQLDatabase(structured_engine)
    execute_query_tool = QuerySQLDatabaseTool(db=sql_db_structured)

    return sql_db_structured
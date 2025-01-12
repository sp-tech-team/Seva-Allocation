import pandas as pd
from training_data import  create_participant_db_df

input_participant_info_csv = "data/input_participant_info_raw.csv"

target_columns = ['SP ID', 'Work Experience/Company', 'Work Experience/Designation',
    'Work Experience/Tasks', 'Work Experience/Industry',
    'Education/Qualifications', 'Education/Specialization',
    'Any Additional Skills', 'Computer Skills', 'Skills',
    'Languages', 'Gender', 'Age', 'Work Experience/From Date', 'Work Experience/To Date']
participant_info_raw_df = pd.read_csv(input_participant_info_csv)
participant_db_df = create_participant_db_df(participant_info_raw_df, target_columns)
context = " ".join(participant_db_df["summary"])
with open("context.txt", "w") as text_file:
    text_file.write(context)
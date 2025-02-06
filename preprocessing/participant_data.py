import pandas as pd
import os
import pdb
from .concat_participant_features import ConcatTool
from datetime import datetime

def clean_participant_data(participant_info_df, target_columns, columns_to_concatenate):
    """
    Clean the participant data.

    Args:
        participant_info_df (pd.DataFrame): DataFrame containing participant info.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    columns_to_fill = list(filter(lambda x: x not in columns_to_concatenate, target_columns))
    participant_cleaned_df = ConcatTool.concat_target_cols(participant_info_df, columns_to_fill, columns_to_concatenate, "SP ID")
    return participant_cleaned_df[target_columns]

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%Y-%m-%d')
    except ValueError:
        return None

def create_participant_summary(row):
    # Combine Experience and Experience_Tasks into sentences
    experience_sentences = []
    zip_cols = [row['Work Experience/Industry'], row['Work Experience/Tasks'], row['Work Experience/From Date'], row['Work Experience/To Date']]
    for exp, task, start_date_s, end_date_s in zip(*zip_cols):
        try:
            start_date = parse_date(start_date_s)
            end_date = parse_date(end_date_s)
            if start_date and end_date:
                months_of_experience = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
                experience_sentence = f"A {exp} in the industry doing {task} tasks for {months_of_experience} months."
            else:
                experience_sentence = f"A {exp} in the industry doing {task} tasks."
        except Exception as e:
            experience_sentence = f"A {exp} in the industry doing {task} tasks."
        experience_sentences.append(experience_sentence)
    summary = f"""
Participant {row['SP ID']} has the following work experience: {' '.join(experience_sentences)}
They have the following qualifications: {row['Education/Qualifications']} in {row['Education/Specialization']}.
They have the following skills: {row['Skills']} and {row['Any Additional Skills']} and computer skill 
{row['Computer Skills']} and know the following languages: {'.'.join(row['Languages'])}.
The participant is a {row['Gender']} and {row['Age']} years old.\n"""
    return summary

def create_participant_db_df(participant_info_raw_df, target_columns):
    """
    Create a DataFrame from the participant info csv.

    Args:
        participant_info_raw_df (pd.DataFrame): DataFrame containing participant info.

    Returns:
        pd.DataFrame: DataFrame containing participant info.
    """
    columns_to_concatenate = ['Languages', 'Work Experience/Industry', 'Work Experience/Tasks', 'Work Experience/From Date', 'Work Experience/To Date']
    participant_info_df = clean_participant_data(participant_info_raw_df, target_columns=target_columns, columns_to_concatenate=columns_to_concatenate)
    participant_info_df['SP ID'] = participant_info_df['SP ID'].astype(int)

    from_date = 'Work Experience/From Date'
    to_date = 'Work Experience/To Date'
    len_mask = participant_info_df[from_date].apply(len) != participant_info_df[to_date].apply(len)
    na_mask = participant_info_df.apply(lambda row: "NA" in row[from_date] or "NA" in row[to_date], axis=1)
    mask = len_mask | na_mask
    participant_info_df.loc[mask, from_date] = participant_info_df.loc[mask, from_date].apply(lambda x: ["NA"])
    participant_info_df.loc[mask, to_date] = participant_info_df.loc[mask, to_date].apply(lambda x: ["NA"])
    participant_info_df = participant_info_df[target_columns]
    participant_info_df["summary"] = participant_info_df.apply(create_participant_summary, axis=1)
    return participant_info_df


if __name__ == "__main__":
    target_columns = ['SP ID', 'Work Experience/Company', 'Work Experience/Designation',
    'Work Experience/Tasks', 'Work Experience/Industry',
    'Education/Qualifications', 'Education/Specialization',
    'Any Additional Skills', 'Computer Skills', 'Skills',
    'Languages', 'Gender', 'Age', 'Work Experience/From Date', 'Work Experience/To Date']
    participant_info_raw_df = pd.read_csv('../../data/input_participant_info_raw.csv')
    participant_db_df = create_participant_db_df(participant_info_raw_df, target_columns)
    print(participant_db_df)
    participant_db_df.to_csv("../../data/input_participant_info_cleaned.csv")
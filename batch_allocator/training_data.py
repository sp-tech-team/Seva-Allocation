import pandas as pd
import os
from concatenation.Concatenation_Library import Concatenation_Handler
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
    participant_cleaned_df = Concatenation_Handler.Concatenation_Main_Using_Local_Downloaded_File(participant_info_df, columns_to_fill, columns_to_concatenate, "SP ID")
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
    participant_info_df = participant_info_df[target_columns]
    participant_info_df['SP ID'] = participant_info_df['SP ID'].astype(int)
    participant_info_df["summary"] = participant_info_df.apply(create_participant_summary, axis=1)
    return participant_info_df

def clean_vrf_data(vrf_df):
    """
    Clean the VRF data.

    Args:
        vrf_df (pd.DataFrame): DataFrame containing VRF data.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    columns_to_fill = [
            "Request Name",
            "Job Title",
            "Job Description",
            "Department",
            ]
    columns_to_concatenate = [
        "Languages",
        "Skills/Keywords",
        "Add'l Skills"
        ]
    vrf_cleaned_df = Concatenation_Handler.Concatenation_Main_Using_Local_Downloaded_File(vrf_df, columns_to_fill, columns_to_concatenate, "Request Name")
    return vrf_cleaned_df

def create_vrf_specific_summary(row):
    summary =  f"""
    The job titled: "{row['Job Title']}" has a description: {row['Job Description']},
    and expects the following skills: {', '.join(row['Skills/Keywords'])}, and {', '.join(row["Add'l Skills"])}
    and know the languages: {', '.join(row['Languages'])}.
    """
    return summary

def create_vrf_generic_summary(row):
    summary = f"""
    The job titled: "{row['Job Title']}" has a description: {row['Job Title Generic Description']}.
    """
    return summary

def create_vrf_db_df(vrf_raw_df, generic_jobs_df):
    required_columns = ['Request Name', 'Job Title', 'Job Description', 'Department']
    columns = required_columns + ['Skills/Keywords', 'Add\'l Skills', 'Languages']
    vrf_df = clean_vrf_data(vrf_raw_df)
    vrf_df = vrf_df[columns].copy()
    vrf_df = vrf_df.dropna(subset=required_columns)
    vrf_df = vrf_df[columns].fillna("NA")
    vrf_df = vrf_df.apply(lambda x: x.replace("\n", " "))
    # Make specific training data
    vrf_specific_df = vrf_df[["Job Title", "Request Name", "Department"]].copy()
    vrf_specific_df["summary"] = vrf_df.apply(create_vrf_specific_summary, axis=1)
    # Make generic training data
    generic_df = generic_jobs_df[["Job Title"]].copy()
    generic_df["Request Name"] = ""
    generic_df['Department'] = ""
    generic_df["summary"] = generic_jobs_df.apply(create_vrf_generic_summary, axis=1)

    # Combine the two dataframes
    target_columns = ['Job Title', 'Request Name', 'Department', 'summary']
    return pd.concat([vrf_specific_df[target_columns], generic_df[target_columns]]).reset_index(drop=True)



def create_vrf_training_data(vrf_df,
                             generic_jobs_df,
                             output_dir):
    """
    Create training data for VRF model.

    Args:
        vrf_df (pd.DataFrame): DataFrame containing VRF data.
        output_dir (str): Output directory to save training data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    required_columns = ['Request Name', 'Job Title', 'Job Description', 'Department', 'Skills/Keywords']
    columns = required_columns + ['Add\'l Skills', 'Languages']
    vrf_df = vrf_df.dropna(subset=required_columns)
    vrf_df = vrf_df[columns].fillna("NA")
    vrf_df = vrf_df.apply(lambda x: x.replace("\n", " "))
    # Make specific training data
    vrf_specific_df = vrf_df[["Job Title", "Request Name", "Department"]]
    vrf_specific_df["summary"] = "The job titled: \"" + vrf_df["Job Title"] + "\" has a description: " + vrf_df["Job Description"] + \
        ", and expects the following skills: " + vrf_df["Skills/Keywords"].apply(lambda x: ', '.join(x)) + ", and " + vrf_df["Add'l Skills"].apply(lambda x: ', '.join(x)) + \
        " and know the languages: " + vrf_df["Languages"].apply(lambda x: ', '.join(x)) + "."
    vrf_specific_df.to_csv(os.path.join(output_dir, "vrf_specific_train_data.csv"), index=False)
    # Make generic training data
    generic_df = generic_jobs_df[["Job Title"]].copy()
    generic_df["Request Name"] = ""
    generic_df['Department'] = ""
    generic_df["summary"] = "The job titled: \"" + generic_df["Job Title"] + "\" has a description: " + generic_jobs_df["Job Title Generic Description"] + "."
    generic_df.to_csv(os.path.join(output_dir, "vrf_generic_train_data.csv"), index=False)

def create_vrf_single_df(specific_train_data_file = "",
                         generic_train_data_file = ""):
    """
    Create a single DataFrame from the VRF training data.
    
    Args:
        specific_train_data_file (str): Path to the specific training data file.
        generic_train_data_file (str): Path to the generic training data file.
    """
    dfs = []
    target_columns = ['Job Title', 'Request Name', 'Department', 'summary']
    if specific_train_data_file:
        specific_df = pd.read_csv(specific_train_data_file, header=0).fillna('')
        dfs.append(specific_df[target_columns])
    if generic_train_data_file:
        generic_df = pd.read_csv(generic_train_data_file, header=0).fillna('')
        dfs.append(generic_df[target_columns])
    return pd.concat(dfs).reset_index(drop=True)



def main():
    pass
if __name__ == "__main__":
    main()
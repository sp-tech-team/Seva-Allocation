import pandas as pd
import os
import pdb
from concat_participant_features import ConcatTool
from datetime import datetime

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


class ParticipantData():
    def __init__(self, participant_info_raw_df):
        self.all_columns = ['SP ID',
                            'Work Experience/Company', 'Work Experience/Designation',
                            'Work Experience/Tasks', 'Work Experience/Industry',
                            'Work Experience/From Date', 'Work Experience/To Date',
                            'Education/Qualifications', 'Education/Specialization',
                            'Skills', 'Any Additional Skills', 'Computer Skills', 
                            'Languages', 'Gender', 'Age']
        self.participant_info_raw_df = participant_info_raw_df[self.all_columns]
        self.participant_info_df = None
        self.columns_to_concatenate = ['Work Experience/Company', 'Work Experience/Designation',
                                       'Work Experience/Tasks', 'Work Experience/Industry',
                                       'Work Experience/From Date', 'Work Experience/To Date',
                                       'Education/Qualifications', 'Education/Specialization',
                                       'Skills','Languages']
        self.concat_fill_str = 'NA'
    
    def clean_participant_data(self):
        """
        Clean the participant data.

        Args:
            participant_info_df (pd.DataFrame): DataFrame containing participant info.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        participant_info_df = self.participant_info_raw_df.copy()

        # Roll up columns in columns_to_concatenate into a single cell entry as a list
        columns_to_fill = list(filter(lambda x: x not in self.columns_to_concatenate, self.all_columns))
        participant_info_df = ConcatTool.concat_target_cols(participant_info_df,
                                                               columns_to_fill,
                                                               self.columns_to_concatenate,
                                                               "SP ID",
                                                               fill_str=self.concat_fill_str)
        # Convert SP ID to int
        participant_info_df['SP ID'] = participant_info_df['SP ID'].astype(int)
        # Clean up Experience Date columns that are inconsistent
        from_date = 'Work Experience/From Date'
        to_date = 'Work Experience/To Date'
        len_mask = participant_info_df[from_date].apply(len) != participant_info_df[to_date].apply(len)
        na_mask = participant_info_df.apply(lambda row: "NA" in row[from_date] or "NA" in row[to_date], axis=1)
        mask = len_mask | na_mask
        participant_info_df.loc[mask, from_date] = participant_info_df.loc[mask, from_date].apply(lambda x: ["NA"])
        participant_info_df.loc[mask, to_date] = participant_info_df.loc[mask, to_date].apply(lambda x: ["NA"])
        return participant_info_df
    
    def create_years_of_experience_col(self, participant_info_df):
        from_col, to_col = "Work Experience/From Date", "Work Experience/To Date"
        results = []
        for _, row in participant_info_df.iterrows():
            if len(row[from_col]) != len(row[to_col]) or not len(row[from_col]) or self.concat_fill_str in row[from_col] or self.concat_fill_str in row[to_col]:
                results.append(None)
                continue
            durations = []
            for start, end in zip(row[from_col], row[to_col]):
                s_date, e_date = parse_date(start), parse_date(end)
                if s_date and e_date:
                    durations.append(round((e_date - s_date).days / 365, 2))
                else:
                    durations.append(None)
            results.append(durations)
        return pd.Series(results)

    def create_participant_info_df(self):
        """
        Create a DataFrame from the participant info csv.

        Args:
            participant_info_raw_df (pd.DataFrame): DataFrame containing participant info.

        Returns:
            pd.DataFrame: DataFrame containing participant info.
        """
        participant_info_df = self.clean_participant_data()
        participant_info_df["Years of Experience"] = self.create_years_of_experience_col(participant_info_df)
        participant_info_df["summary"] = participant_info_df.apply(create_participant_summary, axis=1)
        return participant_info_df


if __name__ == "__main__":
    participant_info_raw_df = pd.read_csv('../data/input_participant_info_raw.csv')
    participant_data = ParticipantData(participant_info_raw_df)
    participant_info_df = participant_data.create_participant_info_df()
    participant_info_df.to_csv("../data/input_participant_info_cleaned.csv")
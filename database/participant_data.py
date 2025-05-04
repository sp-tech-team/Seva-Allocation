import pandas as pd
import json
import pdb
from database.concat_participant_features import ConcatTool
from datetime import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--participant_info_raw_csv',
        type=str,
        help='Raw participant info csv file from Sadhaka',
        default='data/input_participant_info_raw.csv'
    )
    parser.add_argument(
        '--output_participant_info_cleaned_csv',
        type=str,
        help='Output cleaned participant info csv file',
        default='data/input_participant_info_cleaned.csv'
    )
    parser.add_argument(
        '--column_info_config_json',
        type=str,
        help='JSON file for column info configuration',
        default='database/column_info_config.json'
    )
    
    return parser.parse_args()

def filter_data_columns(data_columns: dict, *,
                        data_mode: str = None,
                        derivative_column: bool = None,
                        needs_roll_up: bool = None,
                        upload_db: bool = None) -> list[str]:
    filtered_cols = []
    for col_name, props in data_columns.items():
        if (
            (data_mode is None or props.get("data_mode") == data_mode) and
            (derivative_column is None or props.get("derivative_column") == derivative_column) and
            (needs_roll_up is None or props.get("needs_roll_up") == needs_roll_up) and
            (upload_db is None or props.get("upload_db") == upload_db)
        ):
            filtered_cols.append(col_name)
    return filtered_cols

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
    def __init__(self, participant_info_raw_df, column_info_config):
        self.column_info_config = column_info_config
        self.all_columns = [self.column_info_config["data_key_column"]] + filter_data_columns(self.column_info_config["data_columns"], derivative_column=False)
        self.participant_info_raw_df = participant_info_raw_df[self.all_columns]
        self.columns_to_concatenate = filter_data_columns(self.column_info_config["data_columns"], derivative_column=False, needs_roll_up=True)
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
                                                              self.column_info_config["data_key_column"],
                                                               fill_str=self.concat_fill_str)
        # Convert SP ID to int
        participant_info_df[self.column_info_config["data_key_column"]] = participant_info_df[self.column_info_config["data_key_column"]].astype(int)
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

    def create_total_years_experience_col(self, participant_info_df):
        experience_arrays = self.create_years_of_experience_col(participant_info_df)
        total_experience = experience_arrays.apply(lambda arr: sum(filter(None, arr)) if isinstance(arr, list) else 0) #Handle None values, and non list values.
        return total_experience

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
        participant_info_df["Total Years of Experience"] = self.create_total_years_experience_col(participant_info_df)
        participant_info_df["Summary"] = participant_info_df.apply(create_participant_summary, axis=1)
        pref_col_order = [column_info_config["data_key_column"]] + list(column_info_config["data_columns"].keys())
        participant_info_df = participant_info_df[pref_col_order]
        return participant_info_df


if __name__ == "__main__":
    args = parse_args()
    with open(args.column_info_config_json, 'r') as f:
        column_info_config = json.load(f)
    participant_info_raw_df = pd.read_csv(args.participant_info_raw_csv)
    participant_data = ParticipantData(participant_info_raw_df, column_info_config)
    participant_info_df = participant_data.create_participant_info_df()
    print("writing cleaned file")
    participant_info_df.to_csv(args.output_participant_info_cleaned_csv, index=False)
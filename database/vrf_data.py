import pandas as pd
from preprocessing.concat_participant_features import ConcatTool

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
    vrf_cleaned_df = ConcatTool.concat_target_cols(vrf_df, columns_to_fill, columns_to_concatenate, "Request Name")
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


class VrfData():
    def __init__(self, vrf_raw_df, generic_jobs_df):
        self.vrf_raw_df = vrf_raw_df
        self.generic_jobs_df = generic_jobs_df
    
    def create_vrf_info_df(self):
        required_columns = ['Request Name', 'Job Title', 'Job Description', 'Department']
        columns = required_columns + ['Skills/Keywords', 'Add\'l Skills', 'Languages']
        vrf_df = clean_vrf_data(self.vrf_raw_df)
        vrf_df = vrf_df[columns].copy()
        vrf_df = vrf_df.dropna(subset=required_columns)
        vrf_df = vrf_df[columns].fillna("NA")
        vrf_df = vrf_df.apply(lambda x: x.replace("\n", " "))
        # Make specific training data
        vrf_specific_df = vrf_df[["Job Title", "Request Name", "Department"]].copy()
        vrf_specific_df["summary"] = vrf_df.apply(create_vrf_specific_summary, axis=1)
        # Make generic training data
        generic_df = self.generic_jobs_df[["Job Title"]].copy()
        generic_df["Request Name"] = ""
        generic_df['Department'] = ""
        generic_df["summary"] = self.generic_jobs_df.apply(create_vrf_generic_summary, axis=1)

        # Combine the two dataframes
        target_columns = ['Job Title', 'Request Name', 'Department', 'summary']
        return pd.concat([vrf_specific_df[target_columns], generic_df[target_columns]]).reset_index(drop=True)

    
        
import pandas as pd
import os

def create_participant_db_df(participant_info_csv):
    """
    Create a DataFrame from the participant info csv.

    Args:
        participant_info_csv (str): Path to the participant info csv.

    Returns:
        pd.DataFrame: DataFrame containing participant info.
    """
    participant_info = pd.read_csv(participant_info_csv)
    
    return 

def create_vrf_training_data(vrf_df,
                             output_dir):
    """
    Create training data for VRF model.

    Args:
        vrf_df (pd.DataFrame): DataFrame containing VRF data.
        output_dir (str): Output directory to save training data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    required_columns = ['Job Title', 'Department', 'Job Title Generic Description', 'Request Name', 'Request Name Description', 'Skills/Keywords']
    columns = required_columns + ['Add\'l Skills', 'Languages']
    vrf_df = vrf_df.dropna(subset=required_columns)
    vrf_df = vrf_df[columns].fillna("NA")
    vrf_df = vrf_df.apply(lambda x: x.replace("\n", " "))
    vrf_df["Request Id"] = vrf_df["Request Name"].str.split(" - ", n=1).str[1]
    # Make generic training data
    vrf_generic_df = vrf_df[["Job Title"]].drop_duplicates(subset=["Job Title"])
    vrf_generic_df["Request Name"] = ""
    vrf_generic_df['Department'] = ""
    vrf_generic_df["summary"] = "The job titled: \"" + vrf_generic_df["Job Title"] + "\" has a description: " + vrf_df["Job Title Generic Description"] + "."
    vrf_generic_df.to_csv(os.path.join(output_dir, "vrf_generic_train_data.csv"), index=False)
    # Make specific training data
    vrf_specific_df = vrf_df[["Job Title", "Request Name", "Department"]]
    vrf_specific_df["summary"] = "The job titled: \"" + vrf_df["Job Title"] + "\" has a description: " + vrf_df["Request Name Description"] + \
        ", and expects the following skills: " + vrf_df["Skills/Keywords"] + ", and " + vrf_df["Add'l Skills"] + \
        " and know the languages: " + vrf_df["Languages"] + "."
    vrf_specific_df.to_csv(os.path.join(output_dir, "vrf_specific_train_data.csv"), index=False)

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
        specific_df = pd.read_csv(specific_train_data_file, header=0)
        dfs.append(specific_df[target_columns])
    if generic_train_data_file:
        generic_df = pd.read_csv(generic_train_data_file, header=0)
        dfs.append(generic_df[target_columns])
    
    return pd.concat(dfs).reset_index(drop=True)


def main():
    vrf_df = pd.read_csv('data/vrf_data.csv')
    create_vrf_training_data(vrf_df, 'data/generated_training_data/')
    print("Training Data Successfully created.")
if __name__ == "__main__":
    main()
import pandas as pd
import os

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
    required_columns = ['Job Title', 'Job Title Generic Description', 'Request Name', 'Request Name Description', 'Skills/Keywords']
    columns = required_columns + ['Add\'l Skills', 'Languages']
    vrf_df = vrf_df.dropna(subset=required_columns)
    vrf_df = vrf_df[columns].fillna("NA")
    vrf_df = vrf_df.apply(lambda x: x.replace("\n", " "))
    vrf_df["Request Id"] = vrf_df["Request Name"].str.split(" - ", n=1).str[1]
    # Make generic training data
    vrf_generic_df = vrf_df[["Job Title", "Job Title Generic Description"]].drop_duplicates(subset=["Job Title", "Job Title Generic Description"])
    vrf_generic_df["summary"] = "The job titled: \"" + vrf_generic_df["Job Title"] + "\" has a description: " + vrf_generic_df["Job Title Generic Description"] + "."
    vrf_generic_df.to_csv(os.path.join(output_dir, "vrf_generic_train_data.csv"), index=False)
    # Make specific training data
    vrf_specific_df = vrf_df[["Request Id"]]
    vrf_specific_df["summary"] = "A specific job request for the job titled: \"" + vrf_df["Job Title"] + "\" has a request id [" + \
        vrf_df["Request Id"] + "] with a specific request description: " + vrf_df["Request Name Description"] + \
        ", and requests the following skills: " + vrf_df["Skills/Keywords"] + ", and " + vrf_df["Add'l Skills"] + \
        " and know the languages: " + vrf_df["Languages"] + "."
    vrf_specific_df.to_csv(os.path.join(output_dir, "vrf_specific_train_data.csv"), index=False)

def create_vrf_single_txt_corpus(specific_train_data_file = "",
                                 generic_train_data_file = ""):
    """
    Create a single text corpus from the VRF training data.
    
    Args:
        specific_train_data_file (str): Path to the specific training data file.
        generic_train_data_file (str): Path to the generic training data file.
    """
    summaries = [""]
    if specific_train_data_file != "":
        specific_df = pd.read_csv(specific_train_data_file, header=0)
        summaries += specific_df["summary"].tolist()
    if generic_train_data_file != "":
        generic_df = pd.read_csv(generic_train_data_file, header=0)
        summaries += generic_df["summary"].tolist()
    return "\n".join(summaries)

def create_vrf_single_df(specific_train_data_file = "",
                         generic_train_data_file = ""):
    """
    Create a single DataFrame from the VRF training data.
    
    Args:
        specific_train_data_file (str): Path to the specific training data file.
        generic_train_data_file (str): Path to the generic training data file.
    """
    dfs = []
    target_columns = ['Job Title', 'summary']
    if specific_train_data_file != "":
        specific_df = pd.read_csv(specific_train_data_file, header=0)
        specific_df = specific_df.rename(columns={'Request Id': 'Job Title'})
        dfs.append(specific_df[target_columns])
    if generic_train_data_file != "":
        generic_df = pd.read_csv(generic_train_data_file, header=0)
        dfs.append(generic_df[target_columns])
    
    return pd.concat(dfs).reset_index(drop=True)


def main():
    vrf_df = pd.read_csv('data/vrf_data.csv')
    create_vrf_training_data(vrf_df, 'data/generated_training_data/')
    corpus = create_vrf_single_txt_corpus('data/generated_training_data/vrf_specific_train_data.csv', 'data/generated_training_data/vrf_generic_train_data.csv')
    print("Training Data Successfully created.")
if __name__ == "__main__":
    main()
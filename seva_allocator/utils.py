import pickle
import os
from datetime import datetime
from llama_index.core import StorageContext, load_index_from_storage
import pandas as pd

INDEX_DIR_PREFIX = "index-"
RESULTS_DIR_PREFIX = "results-"
RESULTS_FILE_NAME = "results.csv"

def get_index_version(base_dir, version='latest'):
    """
    Finds the requested version of timestamped Index directories, pass 'latest' for getting the
    most recent index.

    Args:
        base_dir (str): Path to the base directory containing timestamped directories.
        version (str): The version of the index to retrieve. Default is 'latest'. Format is 'YYYYMMDD_HHMMSS'.

    Returns:
        str: The path to the latest version directory or None if no directories are found.
    """
    # List all subdirectories in the base directory
    subdirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(INDEX_DIR_PREFIX)
    ]
    print("subdirs:", subdirs)
    
    if not subdirs:
        print("No Index directories found.")
        return None

    # Extract timestamps from directory names
    timestamped_dirs = []
    for subdir in subdirs:
        try:
            timestamp_str = subdir.split("-")[1]  # Extract timestamp part
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            timestamped_dirs.append((timestamp, subdir))
        except (IndexError, ValueError):
            print(f"Skipping invalid directory name: {subdir}")

    if not timestamped_dirs:
        print("No valid timestamped directories found.")
        return None

    target_dir = ""
    # Find the directory with the latest timestamp
    if version == 'latest':
        timestamp, target_dir = max(timestamped_dirs, key=lambda x: x[0])
    else:
        matched_dirs = [dir for dir in timestamped_dirs if dir[0].strftime("%Y%m%d_%H%M%S") == version]
        if matched_dirs:
            timestamp, target_dir = matched_dirs[0]
        else:
            raise ValueError(f"No directory found for version: {version}")
    return os.path.join(base_dir, target_dir)

def create_timestamped_index(base_dir, index):
    """
    Creates a timestamped directory for a Index and saves the graph.

    Args:
        base_dir (str): The base directory where timestamped directories are created.
        index (BaseIndex): The Index to save.

    Returns:
        str: Path to the newly created directory.
    """
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the directory name
    folder_name = INDEX_DIR_PREFIX + timestamp
    folder_path = os.path.join(base_dir, folder_name)

    # Create the directory
    os.makedirs(folder_path, exist_ok=True)

    # Save the graph to the directory
    index.storage_context.persist(persist_dir=folder_path)
    print(f"Index saved to: {folder_path}")
    return folder_path

def create_timestamped_results(base_dir, results_df):
    """
    Creates a timestamped directory for a list of results and saves them to a pickle file.

    Args:
        base_dir (str): The base directory where timestamped directories are created.
        results_df (pd.DataFrame): The DataFrame of results to save.

    Returns:
        str: Path to the newly created directory.
    """
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the directory name
    folder_name = RESULTS_DIR_PREFIX + timestamp
    folder_path = os.path.join(base_dir, folder_name)

    # Create the directory
    os.makedirs(folder_path, exist_ok=True)

    results_df.to_csv(os.path.join(folder_path, RESULTS_FILE_NAME), index=False)
    print(f"Results saved to: {folder_path}")
    return folder_path

def extract_jobs_from_nodes(nodes, job_list):
    """
    Extract jobs from the nodes and return the matched jobs.
    
    Args:
        nodes (List[NodeWithScore]): The nodes to extract jobs from.
        job_list (List[str]): The list of job titles to search for in the nodes.
    
    Returns:
        Set[str]: The set of matched job titles.
    """
    matched_jobs = set()
    for node in nodes:
        for job in job_list:
            if job in node.get_content():
                matched_jobs.add(job)
                break  # Avoid duplicate checks for the same node
    return list(matched_jobs)


def load_cached_indexes(pg_store_base_dir = "", vector_store_base_dir = "", pg_version="latest", vector_version="latest"):
    """
    Load the cached Property Graph and Vector indexes.
    
    Args:
        pg_store_dir (str): The directory path to the Property Graph index.
        vector_store_dir (str): The directory path to the Vector index.
        pg_version (str): The version of the Property Graph index to load.
        vector_version (str): The version of the Vector index to load.
    
    Returns:
        Tuple: The loaded Property Graph index, Vector index, Property Graph store directory, and Vector store directory.
    """
    if pg_store_base_dir:
        print("Loading cached Property Graph Index...")
        pg_store_dir = get_index_version(pg_store_base_dir, version=pg_version)
        pg_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=pg_store_dir))
    else:
        pg_index = None
        pg_store_dir = ""

    print("Loading cached Vector Index...")
    if vector_store_base_dir:
        vector_store_dir = get_index_version(vector_store_base_dir, version=vector_version)
        storage_context = StorageContext.from_defaults(persist_dir=vector_store_dir)
        vector_index = load_index_from_storage(storage_context)
    else:
        vector_index = None
        vector_store_dir = ""
    return pg_index, vector_index, pg_store_dir, vector_store_dir

def get_depts_from_job(job_title, vrf_df):
    return vrf_df[vrf_df['Job Title'] == job_title]['Department'].drop_duplicates().values

def get_depts_from_job_df(results_df, vrf_df, pred_column_prefix="Predicted Rank"):
    """
    Get the departments for the predicted jobs in the results dataframe.
    
    Args:
        results_df (pd.DataFrame): The results dataframe.
        vrf_df (pd.DataFrame): The vrf dataframe.
    
    Returns:
        depts (pd.Series): A series with the departments for the predicted jobs.
    """
    predicted_columns = [col for col in results_df.columns if col.startswith(pred_column_prefix)]
    def get_depts(row):
        depts = ""
        for job_title in row[predicted_columns]:
            if pd.isna(job_title):
                depts += ",NA, "
            else:
                job_depts = get_depts_from_job(job_title, vrf_df)
                depts += job_title + ": " + ", ".join(job_depts) + " "
        return depts
    depts = results_df.apply(get_depts, axis=1)
    return depts
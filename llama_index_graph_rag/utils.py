import pickle
import os
from datetime import datetime
from llama_index.core.indices.base import BaseIndex

INDEX_DIR_PREFIX = "index-"
RESULTS_DIR_PREFIX = "results-"
RESULTS_FILE_NAME = "eval_results.csv"

def get_latest_index_version(base_dir):
    """
    Finds the latest version of timestamped Index directories.

    Args:
        base_dir (str): Path to the base directory containing timestamped directories.

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

    # Find the directory with the latest timestamp
    latest_timestamp, latest_dir = max(timestamped_dirs, key=lambda x: x[0])

    return os.path.join(base_dir, latest_dir)


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

def create_timestamped_pg_index(base_dir, pg_index):
    """
    Creates a timestamped directory for a Property Graph Index and saves the graph.

    Args:
        base_dir (str): The base directory where timestamped directories are created.
        pg_index (PropertyGraphIndex): The Property Graph Index to save.

    Returns:
        str: Path to the newly created directory.
    """

    folder_path = create_timestamped_index(base_dir, pg_index)
    pg_index.property_graph_store.save_networkx_graph(name=os.path.join(folder_path, "kg.html"))
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

def save_to_pickle(obj, file_path):
    """Saves a Python object to a pickle file.

    Args:
        obj: The Python object to save.
        file_path: The path to the pickle file.
    """
    try:
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        print(f"Object successfully saved to {file_path}")
    except Exception as e:
        print(f"Failed to save object to {file_path}: {e}")

def load_from_pickle(file_path):
    """Loads a Python object from a pickle file.

    Args:
        file_path: The path to the pickle file.

    Returns:
        The loaded Python object.
    """
    try:
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        print(f"Object successfully loaded from {file_path}")
        return obj
    except Exception as e:
        print(f"Failed to load object from {file_path}: {e}")
        return None
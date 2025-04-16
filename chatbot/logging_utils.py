import os
from datetime import datetime
import pandas as pd

RESULTS_DIR_PREFIX = "results-"
RESULTS_FILE_NAME = "results.json"


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
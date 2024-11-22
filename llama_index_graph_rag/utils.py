import pickle

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

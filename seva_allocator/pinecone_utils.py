
import pinecone
from pinecone import Pinecone
from pinecone import ServerlessSpec


def index_name_exists(pinecone_index_name: str, pc: Pinecone) -> bool:
    """
    Check if a Pinecone index exists.
    
    Args:
        pinecone_index_name (str): The name of the Pinecone index.
        pc (Pinecone): The Pinecone instance.
    
    Returns:
        bool: True if the index exists, False otherwise.
    """
    return pinecone_index_name in [idx_dict["name"] for idx_dict in pc.list_indexes()]

def get_pinecone_index(index_name: str, pc: Pinecone, create_if_not_exists: bool = True) -> pinecone.Index:
    """
    Get a Pinecone index by name or create new one if doesn't exist.
    
    Args:
        index_name (str): The name of the Pinecone index.
        pc (Pinecone): The Pinecone instance.
        create_if_not_exists (bool): Whether to create the index if it does not exist. Default is True.
    
    Returns:
        pinecone.Index: The Pinecone index.

    Raises:
        ValueError: If the index does not exist and create_if_not_exists is False
    """

    if index_name_exists(index_name, pc):
        return pc.Index(index_name)
    if create_if_not_exists:
        pc.create_index(
            index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) )
    else:
        raise ValueError(f"Index {index_name} not found in Pinecone.")
    return pc.Index(index_name)
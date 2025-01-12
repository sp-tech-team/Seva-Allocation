#!/usr/bin/env python3
"""
main.py: A script demonstrating clean code with command-line arguments.

This script performs a simple operation based on user-provided inputs. It follows
the Google Python style guide for clarity and maintainability.

Usage:
    python main.py --input_file=input.txt --output_file=output.txt --verbose
"""

import argparse
import logging
import os
from typing import List
import pandas as pd

from training_data import  create_participant_db_df, clean_participant_data
from pinecone_utils import get_pinecone_index, clear_index

from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core.schema import TextNode
from dotenv import load_dotenv
load_dotenv()

def index_parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        A namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process an input file and save the results to an output file."
    )

    parser.add_argument(
        "--input_participant_info_csv",
        default='data/input_participant_info_raw.csv',
        help="Path to input participants information.",
    )

    parser.add_argument(
        "--input_participant_info_cleaned_csv",
        default='data/input_participant_info_cleaned.csv',
        help="Path to write cleaned participants information.",
    )

    parser.add_argument(
        "--vector_store_base_dir",
        default='participant_vector_store_versions/',
        help="Path to vector store dbs.",
    )

    parser.add_argument(
        "--pinecone_index_name",
        default='participant-test-local',
        help="Path to vector store dbs.",
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',  # Default False
        help="Enable verbose logging. (default is disabled)"
    )

    return parser.parse_args()

def create_index_nodes(participant_db_df: pd.DataFrame, target_columns: List[str]) -> list[TextNode]:
    """
    Create the index nodes from the training data.
    
    Args:
        participant_db_df (pd.DataFrame): DataFrame containing participant information.
    """
    nodes = []
    for i, (_, row) in enumerate(participant_db_df.iterrows()):
        node = TextNode(
            text=row["summary"],
            _id=str(i),
            metadata = {target: row[target] for target in target_columns}
            )
        nodes.append(node)
    return nodes

def main() -> None:
    """Main entry point of the script."""
    args = index_parse_args()

    # Configure logging
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    if not args.verbose:
        logging.disable(logging.CRITICAL)

    logging.info("Script started.")

    # Setup LLM and Embeddings
    llm = OpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    embedding_model = OpenAIEmbedding()
    Settings.llm = llm
    Settings.embed_model = embedding_model

    # Setup training data
    target_columns = ['SP ID', 'Work Experience/Company', 'Work Experience/Designation',
       'Work Experience/Tasks', 'Work Experience/Industry',
       'Education/Qualifications', 'Education/Specialization',
       'Any Additional Skills', 'Computer Skills', 'Skills',
       'Languages', 'Gender', 'Age', 'Work Experience/From Date', 'Work Experience/To Date']
    participant_info_raw_df = pd.read_csv(args.input_participant_info_csv)
    participant_db_df = create_participant_db_df(participant_info_raw_df, target_columns)
    participant_db_df.to_csv("participant_db_df.csv", index=False)
    nodes = create_index_nodes(participant_db_df, target_columns)
    for node in nodes:
        node.embedding = embedding_model.get_text_embedding(node.get_text())

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = get_pinecone_index(args.pinecone_index_name, pc, create_if_not_exists=True)
    clear_index(index)
    vector_store = PineconeVectorStore(index)
    vector_store.add(nodes)

    logging.info("Script finished.")


if __name__ == "__main__":
    main()

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
import pandas as pd

from training_data import create_vrf_db_df
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
        "--vrf_data_raw_csv",
        default='data/vrf_data_raw.csv',
        help="Path to the vrf csv file.",
    )

    parser.add_argument(
        "--vrf_data_cleaned_out_csv",
        default='data/vrf_data_cleaned.csv',
        help="Path to write the cleaned vrf csv file.",
    )

    parser.add_argument(
        "--generic_jobs_csv",
        default='data/generic_jobs.csv',
        help="Path to the generic_jobs csv file.",
    )

    parser.add_argument(
        "--pinecone_index_name",
        default='vrf-test-local',
        help="Path to vector store dbs.",
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',  # Default False
        help="Enable verbose logging. (default is disabled)"
    )

    return parser.parse_args()

def create_index_nodes(vrf_db_df: pd.DataFrame) -> list[TextNode]:
    """
    Create the index nodes from the training data.
    
    Args:
        vrf_specific_train_data_csv: Path to the specific training data csv file.
        vrf_generic_train_data_csv: Path to the generic training data csv file.
    """
    nodes = []
    for i, (_, row) in enumerate(vrf_db_df.iterrows()):
        if not row["summary"]:
            continue
        node = TextNode(
            text=row["summary"],
            id_=str(i),
            metadata = {
                "Job Title": row["Job Title"],
                "Request Name": row["Request Name"],
                "Department": row["Department"]
            })
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

    print("Creating Vector Store Index...")
    vrf_raw_df = pd.read_csv(args.vrf_data_raw_csv)
    generic_jobs_df = pd.read_csv(args.generic_jobs_csv)
    vrf_db_df = create_vrf_db_df(vrf_raw_df, generic_jobs_df)
    nodes = create_index_nodes(vrf_db_df)
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

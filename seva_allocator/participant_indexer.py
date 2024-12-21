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
import json
import pandas as pd

from utils import create_timestamped_index
from training_data import create_vrf_single_df, create_vrf_training_data

from langchain_openai import OpenAIEmbeddings
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.schema import TextNode
import nest_asyncio
nest_asyncio.apply()
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
        "--vrf_data_csv",
        default='data/vrf_data.csv',
        help="Path to the vrf csv file.",
    )

    parser.add_argument(
        "--generic_jobs_csv",
        default='data/generic_jobs.csv',
        help="Path to the generic_jobs csv file.",
    )

    parser.add_argument(
        "--training_data_dir",
        default='data/generated_training_data/',
        help="Path to the training data directory.",
    )

    parser.add_argument(
        '--setup_training_data',
        action='store_true',
        dest='setup_training_data',  # Default False
        help="Enable setup of training data. (default is disabled)"
    )
    
    parser.add_argument(
        "--vrf_generic_train_data_csv",
        default='data/generated_training_data/vrf_generic_train_data.csv',
        help="Path to the vrf jobs generic training data.",
    )

    parser.add_argument(
        "--vrf_specific_train_data_csv",
        default='data/generated_training_data/vrf_specific_train_data.csv',
        help="Path to the vrf jobs specific training data.",
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',  # Default False
        help="Enable verbose logging. (default is disabled)"
    )

    return parser.parse_args()

def create_index_nodes(vrf_specific_train_data_csv, vrf_generic_train_data_csv):
    """
    Create the index nodes from the training data.
    
    Args:
        vrf_specific_train_data_csv: Path to the specific training data csv file.
        vrf_generic_train_data_csv: Path to the generic training data csv file.
    """
    vrf_single_df = create_vrf_single_df(vrf_specific_train_data_csv, vrf_generic_train_data_csv)
    nodes = []
    for i, (_, row) in enumerate(vrf_single_df.iterrows()):
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

    # Setup training data
    if args.setup_training_data:
        vrf_df = pd.read_csv(args.vrf_data_csv)
        generic_jobs_df = pd.read_csv(args.generic_jobs_csv)
        create_vrf_training_data(vrf_df, generic_jobs_df, 'data/generated_training_data/')

    # Setup LLM and Embeddings
    llm = OpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    embeddings = OpenAIEmbeddings()
    Settings.llm = llm
    Settings.embed_model = embeddings

    print("Creating Vector Store Index...")
    nodes = create_index_nodes(args.vrf_specific_train_data_csv, args.vrf_generic_train_data_csv)
    vector_index = VectorStoreIndex(nodes)
    create_timestamped_index("./vector_store_versions", vector_index)
    logging.info("Script finished.")


if __name__ == "__main__":
    main()

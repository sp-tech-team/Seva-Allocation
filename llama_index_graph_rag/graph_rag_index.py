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
import sys
from pathlib import Path
import json

from graph_rag_lib import GraphRagRetriever
from utils import create_timestamped_index, create_timestamped_pg_index
from training_data import create_vrf_single_df, create_vrf_single_txt_corpus

import nest_asyncio
nest_asyncio.apply()
from dotenv import load_dotenv
# Load environment variables from a .env file
load_dotenv()
# from IPython.display import Markdown, display
from pyvis.network import Network

import os
import pandas as pd

from langchain_openai import OpenAIEmbeddings

from llama_index.llms.openai import OpenAI
from llama_index.core import PropertyGraphIndex, VectorStoreIndex
from llama_index.core import Document
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor, SchemaLLMPathExtractor
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.core.schema import TextNode

# Use typing_extention for forward compatibility with dynamic TypeAlias/ Literal
#from typing import Literal # Python 3.11 and up
from typing_extensions import Literal # Python 3.10 and below



def index_parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        A namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process an input file and save the results to an output file."
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
        "--index_config_json",
        default='configs/index_config.json',
        help="Path index generator configs.",
    )

    parser.add_argument(
        "--pg_index_metadata",
        default='storage',
        help="Info to add contexted to versioned index's.",
    )

    parser.add_argument(
        '--no-vector_db',
        action='store_false',
        dest='update_vector_db',  # Default True
        help="Disable updating vector db (default is enabled)"
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',  # Default False
        help="Enable verbose logging. (default is disabled)"
    )

    return parser.parse_args()

def get_pg_extractor(index_config_json, llm):
    with open(index_config_json, "r") as file:
        index_config = json.load(file)
    entities = Literal[tuple(index_config['property_graph_schema_extractor']['entities'])]
    relations = Literal[tuple(index_config['property_graph_schema_extractor']['relations'])]
    schema = index_config['property_graph_schema_extractor']['schema']
    kg_extractor = SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=entities,
        possible_relations=relations,
        kg_validation_schema=schema,
        strict=False,  # if false, will allow triplets outside of the schema
        num_workers=4,
        max_triplets_per_chunk=10,
    )
    return kg_extractor

def create_pg_index(documents, kg_extractor, llm, embeddings):
    try:
        pg_index = PropertyGraphIndex.from_documents(
            documents,
            llm = llm,
            embed_model = embeddings,
            show_progress=True,
            kg_extractors=[kg_extractor],
        )
    except AssertionError as e:
        print ("Preperty Graph Schema Failed: ", e)
        print("Creating Property Graph Index without Schema...")
        pg_index = PropertyGraphIndex.from_documents(
            documents,
            llm = llm,
            embed_model = embeddings,
            show_progress=True,
        )
    return pg_index

def create_index_nodes(vrf_specific_train_data_csv, vrf_generic_train_data_csv):
    vrf_single_df = create_vrf_single_df(vrf_specific_train_data_csv, vrf_generic_train_data_csv)
    nodes = []
    for i, sentence in enumerate(vrf_single_df['summary']):
        node = TextNode(text=sentence, id_=str(i))
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

    llm = OpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    embeddings = OpenAIEmbeddings()
    Settings.llm = llm
    Settings.embed_model = embeddings

    print("Creating Property Graph Index...")
    kg_extractor = get_pg_extractor(args.index_config_json, llm)
    jobs_train_corpus = create_vrf_single_txt_corpus(specific_train_data_file = "",
                                                     generic_train_data_file = args.vrf_generic_train_data_csv)
    documents = [Document(text=jobs_train_corpus)]
    pg_index = create_pg_index(documents, kg_extractor, llm, embeddings)
    create_timestamped_pg_index("./pg_store_versions", pg_index)

    if args.update_vector_db:
        print("Creating Vector Store Index...")
        nodes = create_index_nodes(args.vrf_specific_train_data_csv, args.vrf_generic_train_data_csv)
        vector_index = VectorStoreIndex(nodes)
        create_timestamped_index("./vector_store_versions", vector_index)
    logging.info("Script finished.")


if __name__ == "__main__":
    main()

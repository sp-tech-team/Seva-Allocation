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



def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        A namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process an input file and save the results to an output file."
    )
    parser.add_argument(
        "--vrf_jobs_train_corpus_txt",
        default='data/generated_training_data/vrf_jobs_train_corpus.txt',
        help="Path to the vrf jobs training corpus.",
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


def main() -> None:
    """Main entry point of the script."""
    args = parse_args()

    # Configure logging
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    if not args.verbose:
        logging.disable(logging.CRITICAL)

    logging.info("Script started.")

    llm = OpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    embeddings = OpenAIEmbeddings()
    Settings.llm = llm
    # Settings.embed_model = embeddings

    with open(args.index_config_json, "r") as file:
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
    print("Creating Property Graph Index...")
    with open(args.vrf_jobs_train_corpus_txt, "r") as file:
        jobs_train_corpus = file.read()
    documents = [Document(text=jobs_train_corpus)]
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
    create_timestamped_pg_index("./pg_store_versions", pg_index)

    if args.update_vector_db:
        print("Creating Vector Store Index...")
        nodes = []
        for i, sentence in enumerate(jobs_train_corpus.split("\n")):
            node = TextNode(text=sentence, id_=str(i))
            nodes.append(node)
        vector_index = VectorStoreIndex(nodes)
        create_timestamped_index("./vector_store_versions", vector_index)
    logging.info("Script finished.")


if __name__ == "__main__":
    main()

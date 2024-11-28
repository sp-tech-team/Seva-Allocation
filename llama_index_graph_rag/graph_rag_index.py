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

from typing import Literal


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
        default='data/vrf_jobs_train_corpus.txt',
        help="Path to the vrf jobs training corpus.",
    )

    parser.add_argument(
        "--vrf_depts_train_corpus_txt",
        default='data/vrf_depts_train_corpus.txt',
        help="Path to the vrf dept training corpus.",
    )

    parser.add_argument(
        "--pg_index_metadata",
        default='storage',
        help="Info to add contexted to versioned index's.",
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Enable verbose logging.",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point of the script."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("Script started.")
    
    with open(args.vrf_jobs_train_corpus_txt, "r") as file:
        jobs_train_corpus = file.read()
    with open(args.vrf_depts_train_corpus_txt, "r") as file:
        dept_train_corpus = file.read()
    documents = [Document(text=jobs_train_corpus + "\n" + dept_train_corpus)]
    # documents = [Document(text=jobs_train_corpus), Document(text=dept_train_corpus)]

    llm = OpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    embeddings = OpenAIEmbeddings()
    Settings.llm = llm
    # Settings.embed_model = embeddings

    entities = Literal["JOB", "SKILL"]#, "DEPARTMENT"]
    relations = Literal["WORKS_WITH", "RELATED_TO", "SIMILAR_TO", "USED_BY"]#, "IS_IN", "HAS"]
    schema = {
        "JOB": ["RELATED_TO", "SIMILAR_TO", "WORKS_WITH"], #, "IS_IN"],
        "SKILL": ["RELATED_TO", "USED_BY"],
        #"DEPARTMENT": ["HAS", "WORKS_WITH"],
    }

    kg_extractor = SchemaLLMPathExtractor(
        llm=llm,
        possible_entities=entities,
        possible_relations=relations,
        kg_validation_schema=schema,
        strict=True,  # if false, will allow triplets outside of the schema
        num_workers=4,
        max_triplets_per_chunk=20,
    )
    print("Creating Property Graph Index...")
    pg_index = PropertyGraphIndex.from_documents(
        documents,
        llm = llm,
        embed_model = embeddings,
        show_progress=True,
      #  kg_extractors=[kg_extractor],
    )
    create_timestamped_pg_index("./pg_store_versions", pg_index)
    vector_index = VectorStoreIndex.from_documents(documents)
    create_timestamped_index("./vector_store_versions", vector_index)
    logging.info("Script finished.")


if __name__ == "__main__":
    main()

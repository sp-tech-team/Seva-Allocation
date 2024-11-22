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

from graph_rag import GraphRagRetriever
from utils import save_to_pickle, load_from_pickle

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
        "--applicant_info_csv",
        default='data/person_to_job.csv',
        help="Path to the applicant info input file.",
    )
    parser.add_argument(
        "--pg_index_pkl",
        default='cache/pg_index.pkl',
        help="Path to the generated Property Graph Index Pickle file.",
    )

    parser.add_argument(
        "--use_cached_index",
        type=bool,
        default=False,
        help="Use a cached index for RAG instead of creating a new one.",
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Enable verbose logging.",
    )

    return parser.parse_args()


def create_applicant_info_corpus(applicant_info_csv: str) -> str:
    """Create a corpus of applicant information from a CSV file.

    Args:
        applicant_info_csv: Path to the CSV file containing applicant information.

    Returns:
        A corpus of applicant information.
    """
    job_assigns_df = pd.read_csv(applicant_info_csv).dropna(subset=['VRF ID']).reset_index(drop=True)#.iloc[:50]
    job_assigns_df['job'] = job_assigns_df['VRF ID'].apply(lambda x: x.split('-')[1])
    job_assigns_df['Skillset'] = job_assigns_df['Skillset'].apply(lambda x: x.replace('\n', ' '))
    job_assigns_df['summary'] = " Participant with skills: " + job_assigns_df['Skillset'] + " was assigned to job: " + job_assigns_df['job']
    corpus = '.'.join(job_assigns_df['summary'])
    return corpus

def main() -> None:
    """Main entry point of the script."""
    args = parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("Script started.")

    llm = OpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    embeddings = OpenAIEmbeddings()
    Settings.llm = llm
    # Settings.embed_model = embeddings

    print("Loading cached Property Graph Index...")
    pg_index = load_index_from_storage(StorageContext.from_defaults(persist_dir="./storage"))
    pg_index.property_graph_store.save_networkx_graph(name="kg.html")
    pg_retriever = pg_index.as_retriever()

    vector_index = VectorStoreIndex.from_documents(documents)
    vector_retriever = VectorIndexRetriever(index=vector_index)

    graph_rag_retriever = GraphRagRetriever(vector_retriever, pg_retriever)

    # create response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
    )

    graph_rag_query_engine = RetrieverQueryEngine(
        retriever=graph_rag_retriever,
        response_synthesizer=response_synthesizer,
    )

    vector_query_engine = vector_index.as_query_engine()

    pg_keyword_query_engine = pg_index.as_query_engine(
        # setting to false uses the raw triplets instead of adding the text from the corresponding nodes
        include_text=False,
        retriever_mode="keyword",
        response_mode="tree_summarize",
    )

    for (query_engine, engine_name) in [(vector_query_engine, "vector"), (pg_keyword_query_engine, "Property Graph"), (graph_rag_query_engine, "Graph RAG")]:
        response = query_engine.query("Give me some information on my data")
        print(engine_name, ": ", response)

    logging.info("Script finished.")


if __name__ == "__main__":
    main()

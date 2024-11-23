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


from graph_rag import GraphRagRetriever
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
        "--applicant_info_csv",
        default='data/person_to_job.csv',
        help="Path to the applicant info input file.",
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


def create_applicant_info_corpus(applicant_info_csv: str) -> str:
    """Create a corpus of applicant information from a CSV file.

    Args:
        applicant_info_csv: Path to the CSV file containing applicant information.

    Returns:
        A corpus of applicant information.
    """
    job_assigns_df = pd.read_csv(applicant_info_csv).dropna(subset=['VRF ID']).reset_index(drop=True).iloc[:50]
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
    
    applicant_info_corpus = create_applicant_info_corpus(args.applicant_info_csv)
    documents = [Document(text=applicant_info_corpus)]

    llm = OpenAI(temperature=0, model_name="gpt-4o", max_tokens=4000)
    embeddings = OpenAIEmbeddings()
    Settings.llm = llm
    # Settings.embed_model = embeddings

    entities = Literal["JOB", "SKILL"]
    relations = Literal["WORKS_WITH", "RELATED_TO", "SIMILAR_TO", "USED_BY"]
    schema = {
        "JOB": ["RELATED_TO", "SIMILAR_TO", "WORKS_WITH"],
        "SKILL": ["RELATED_TO", "USED_BY"],
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
        kg_extractors=[kg_extractor],
    )
    create_timestamped_pg_index("./pg_store_versions", pg_index)
    vector_index = VectorStoreIndex.from_documents(documents)
    create_timestamped_index("./vector_store_versions", vector_index)
    logging.info("Script finished.")


if __name__ == "__main__":
    main()

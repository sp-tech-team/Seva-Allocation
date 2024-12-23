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

from utils import load_cached_indexes

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
import nest_asyncio
nest_asyncio.apply()
from dotenv import load_dotenv
load_dotenv()


def inference_parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        A namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process an input file and save the results to an output file."
    )

    parser.add_argument(
        "--participant_vector_store_base_dir",
        default='participant_vector_store_versions/',
        help="Path to participant vector store dbs.",
    )

    parser.add_argument(
        "--participant_vector_version",
        default='latest',
        help="The version of the index to retrieve. Default is 'latest'. Format is 'YYYYMMDD_HHMMSS'",
    )

    parser.add_argument(
        "--vrf_vector_store_base_dir",
        default='vrf_vector_store_versions/',
        help="Path to vrf vector store dbs.",
    )

    parser.add_argument(
        "--vrf_vector_version",
        default='latest',
        help="The version of the index to retrieve. Default is 'latest'. Format is 'YYYYMMDD_HHMMSS'",
    )

    parser.add_argument(
        "--num_retrievals",
        default=10,
        type=int,
        help="Number of retrievals from db.",
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',  # Default False
        help="Enable verbose logging. (default is disabled)"
    )

    return parser.parse_args()


class CombinedRetriever:
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def retrieve(self, query):
        # Aggregate results from both retrievers
        results = []
        for retriever in self.retrievers:
            results.extend(retriever.retrieve(query))
        return results

db_selector_query = """
The query below is going to be used to ask a question about a private database.
We have two private databases and a RAG model for each of them. Please determine
which database should be used to answer the question. The first database contains
a list of applicants and their qualifications. The second database contains a list
of job postings and their requirements. A query for example that already has a description
of a job with it's requirements and is looking for a participant should only use the
participant private data for the RAG application. On the other hand, if the query is
looking for jobs with a description of a participant, it should only use the job private db.
Please create a resoponse that only says: PARTICIPANT_DB, JOB_DB, or BOTH. The response
you make should have nothing other than the word PARTICIPANT_DB, JOB_DB, or BOTH.  \n
"""

relevance_extractor_query = """
The query below is going to be used to ask a question about a private database.
We have two private databases and a RAG model for each of them. Please extract
sentences from the query are relevant only to the databases. The first database
contains a list of applicants and their qualifications. The second database contains
a list of job postings and their requirements. As an example, there may be sentences
that are relevant to the the llm being queried but would only confuse the database
vector search. For example descriptions of how to formate the output have nothing to
do with participants or jobs in the database. So please extract the sentences that
are relevant to the databases only and only return the extraction text and nothing else.
Here is the query to analyze...
"""

prompt_tmpl = \
"""Context information is below.\n
---------------------\n
{context_str}\n
---------------------\n
Given the context information and not prior knowledge, 
answer the query.\n
Query: {query_str}\n
Answer: 
"""

def process_input(user_input, retrievers, llm):
    relevance_extracted_response = llm.complete(relevance_extractor_query + user_input)
    extracted_user_input = relevance_extracted_response.text
    db_selector_response = llm.complete(db_selector_query + extracted_user_input)
    if db_selector_response.text not in retrievers:
        raise ValueError(f"Invalid database selector response: {db_selector_response.text}")
    nodes = retrievers[db_selector_response.text].retrieve(extracted_user_input)
    context_str = "\n".join(node.get_content() for node in nodes)
    query = prompt_tmpl.format(context_str=context_str, query_str=user_input)
    response = llm.complete(query)
    print(f"LlamaIndex: \n{response}")
    print(f"Relevance Extractor: \n{extracted_user_input}")
    print(f"DB Selector: \n{db_selector_response}")

def main() -> None:
    """Main entry point of the script."""
    args = inference_parse_args()

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

    _, participant_vector_index, _, _ = load_cached_indexes(
        pg_store_base_dir="",
        vector_store_base_dir=args.participant_vector_store_base_dir,
        pg_version="",
        vector_version=args.participant_vector_version)
    participant_vector_retriever = VectorIndexRetriever(
                        index=participant_vector_index,
                        similarity_top_k=args.num_retrievals)

    _, vrf_vector_index, _, _ = load_cached_indexes(
        pg_store_base_dir="",
        vector_store_base_dir=args.vrf_vector_store_base_dir,
        pg_version="",
        vector_version=args.vrf_vector_version)
    vrf_vector_retriever = VectorIndexRetriever(
                        index=vrf_vector_index,
                        similarity_top_k=args.num_retrievals)

    both_vector_retriever = CombinedRetriever([participant_vector_retriever, vrf_vector_retriever])
    retrievers = {
        "PARTICIPANT_DB": participant_vector_retriever,
        "JOB_DB": vrf_vector_retriever,
        "BOTH": both_vector_retriever
    }
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        try:
            process_input(user_input, retrievers, llm)
        except Exception as e:
            print(f"An error occurred: {e}")

    logging.info("Script finished.")


if __name__ == "__main__":
    main()

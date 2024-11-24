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
import pdb


from graph_rag_lib import GraphRagRetriever
from utils import get_latest_index_version

import nest_asyncio
nest_asyncio.apply()
from dotenv import load_dotenv
# Load environment variables from a .env file
load_dotenv()

import os
import pandas as pd

from langchain_openai import OpenAIEmbeddings

from llama_index.llms.openai import OpenAI
from llama_index.core import PropertyGraphIndex, VectorStoreIndex
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        A namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process an input file and save the results to an output file."
    )

    parser.add_argument(
        "--eval_ids_csv",
        default='data/eval_ids.csv',
        help="Path to eval id's.",
    )

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Enable verbose logging.",
    )

    return parser.parse_args()


def run_inference(eval_data: pd.Series, query_engine) -> tuple:
    chunk_size = 20
    chunks = []
    for i in range(0, len(eval_data), chunk_size):
        chunk = eval_data.iloc[i:i + chunk_size]
        corpus = "\n".join(chunk.tolist())
        chunks.append(corpus)
    
    results = []
    for i, chunk in enumerate(chunks):
        query_combined = \
            """
            Could you please assign a job to each of these participants
            using past history of jobs. Please give me the participant's number
            followed by the job title in this format "{Participant_Id}-{Job_Title}\n".
            Please do not add any other text to the response other than the id and title.
            Here is the information about the participants: \n
            """ \
                + chunk
        response = query_engine.query(query_combined)
        lines = response.response.splitlines()
        print(lines)
        for line in lines:
            data_tuple = tuple(line.split("-"))
            results.append(data_tuple)
    return results


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
    Settings.embed_model = embeddings

    print("Loading cached Property Graph Index...")
    latest_pg_store_dir = get_latest_index_version("./pg_store_versions")
    print("directory:", latest_pg_store_dir)
    pg_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=latest_pg_store_dir))
    pg_retriever = pg_index.as_retriever()

    print("Loading cached Vector Index...")
    latest_vector_store_dir = get_latest_index_version("./vector_store_versions")
    storage_context = StorageContext.from_defaults(persist_dir=latest_vector_store_dir)
    vector_index = load_index_from_storage(storage_context)
    vector_retriever = VectorIndexRetriever(index=vector_index)

    print("Making Graph RAG retriever...")
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


    past_participant_df = pd.read_csv('data/past_participant_info.csv')
    eval_ids_df = pd.read_csv('data/eval_ids.csv')
    eval_df = eval_ids_df.merge(past_participant_df, on='Person Id', how='inner')
    eval_df["eval_input"] =  " Participant " + eval_df["Person Id"].astype(str) + " has skills: " + eval_df['Skillset']
    
    results = run_inference(eval_df["eval_input"], graph_rag_query_engine)
    results_df = pd.DataFrame(results, columns=['Person Id', 'Predicted Job Title'])
    results_df["Person Id"] = results_df["Person Id"].astype(int)
    results_df = eval_df[["Person Id", "Skillset","VRF ID"]].merge(results_df, on='Person Id', how='outer')
    results_df['VRF ID'] = results_df['VRF ID'].apply(lambda x: x.split('-')[1])
    print(results_df)
    results_df.to_csv('data/eval_results.csv', index=False)

    logging.info("Script finished.")


if __name__ == "__main__":
    main()

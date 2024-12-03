#!/usr/bin/env python3
"""
main.py: A script demonstrating clean code with command-line arguments.

This script performs a simple operation based on user-provided inputs. It follows
the Google Python style guide for clarity and maintainability.

Usage:
    python main.py --input_file=input.txt --output_file=output.txt --verbose
"""

import argparse
import json
import logging
import pdb


from graph_rag_lib import GraphRagRetriever
from utils import create_timestamped_results, get_latest_index_version

import nest_asyncio
nest_asyncio.apply()
from dotenv import load_dotenv
# Load environment variables from a .env file
load_dotenv()

import os
import pandas as pd

from langchain_openai import OpenAIEmbeddings

from llama_index.llms.openai import OpenAI
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
        "--past_participant_info_csv",
        default='data/past_participant_info.csv',
        help="Path to past participants labeled jobs.",
    )

    parser.add_argument(
        "--vrf_data_csv",
        default='data/vrf_data.csv',
        help="Path to the vrf csv file.",
    )

    parser.add_argument(
        "--prompt_config_json",
        default='configs/prompt_config.json',
        help="Path to json file of prompt query.",
    )

    parser.add_argument(
        "--results_dir",
        default='results/',
        help="Path to model results dir.",
    )

    parser.add_argument(
        "--results_info_file_name",
        default='results_info.txt',
        help="Results info file name.",
    )

    parser.add_argument(
        "--pg_store_dir",
        default='pg_store_versions/',
        help="Path to property graph store dbs.",
    )

    parser.add_argument(
        "--vector_store_dir",
        default='vector_store_versions/',
        help="Path to vector store dbs.",
    )

    parser.add_argument(
        "--num_eval_samples",
        default=20,
        type=int,
        help="Number of past participants to sample for testing.",
    )

    parser.add_argument(
        "--inference_batch_size",
        default=20,
        type=int,
        help="Number of participants to assign in one api request.",
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',  # Default False
        help="Enable verbose logging. (default is disabled)"
    )

    return parser.parse_args()

def build_prompt(prompt_config_file):
    with open(prompt_config_file, 'r') as f:
        prompt_config = json.load(f)
    prompt = ""
    if prompt_config["prompt_mode"] == "COMPLETE":
        prompt_complete_file = prompt_config["prompt_complete_file"]
        with open(prompt_complete_file, 'r') as f:
            prompt = f.read()
    return prompt

def run_inference(eval_data: pd.Series, query_engine, batch_size, prompt) -> tuple:
    chunks = []
    for i in range(0, len(eval_data), batch_size):
        chunk = eval_data.iloc[i:i + batch_size]
        corpus = "\n".join(chunk.tolist())
        chunks.append(corpus)
    
    results = []
    for i, chunk in enumerate(chunks):
        query_combined = prompt + chunk
        response = query_engine.query(query_combined)
        print("Model Response String: ", response)
        lines = response.response.splitlines()
        for line in lines:
            id_and_ranks = line.split("/-/")
            if len(id_and_ranks) > 5:
                print("over 5 predictions")
                pdb.set_trace()
            ranked_jobs = id_and_ranks[1].split(",")
            results.append([id_and_ranks[0]] + ranked_jobs)
    return results

def get_depts_from_job(job_title, vrf_df):
    return vrf_df[vrf_df['Job Title'] == job_title]['Department'].drop_duplicates().values

def get_depts_from_job_df(results_df, vrf_df):
    predicted_columns = [col for col in results_df.columns if col.startswith("Predicted Rank")]
    def get_depts(row):
        depts = ""
        for job_title in row[predicted_columns]:
            if pd.isna(job_title):
                depts += ",NA, "
            else:
                job_depts = get_depts_from_job(job_title, vrf_df)
                print(f"Job Title: {job_title} Depts: {job_depts}")
                depts += job_title + ": " + ", ".join(job_depts) + " "
        return depts
    depts = results_df.apply(get_depts, axis=1)
    return depts

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
    Settings.embed_model = embeddings

    print("Loading cached Property Graph Index...")
    latest_pg_store_dir = get_latest_index_version(args.pg_store_dir)
    print("directory:", latest_pg_store_dir)
    pg_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=latest_pg_store_dir))
    pg_retriever = pg_index.as_retriever()

    print("Loading cached Vector Index...")
    latest_vector_store_dir = get_latest_index_version(args.vector_store_dir)
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


    eval_df = pd.read_csv(args.past_participant_info_csv)
    eval_df = eval_df[eval_df['Seva Allocation Accurate or not']==1].dropna(subset=['Person Id', 'Skillset', 'VRF ID', ]).sample(n=args.num_eval_samples)
    eval_df[["Computer Skills", "Work Designation", "Education", "Education Specialization"]] = eval_df[["Computer Skills", "Work Designation", "Education", "Education Specialization"]].fillna("NA")
    eval_df["eval_input"] =  " Participant " + eval_df["Person Id"].astype(str) + " has skills: " + eval_df['Skillset'] + \
                            "and specifically computer skills: " + eval_df["Computer Skills"] + \
                            ". The participant worked with designation: " + eval_df["Work Designation"] + \
                            "and has a " + eval_df["Education"] + "education specialized in " + eval_df["Education Specialization"]
    prompt = build_prompt(args.prompt_config_json)
    results = run_inference(eval_df["eval_input"], graph_rag_query_engine, args.inference_batch_size, prompt)
    num_ranked_jobs = len(results[0]) - 1
    ranked_job_titles = [f"Predicted Rank {i+1} Job Title" for i in range(num_ranked_jobs)]
    results_df = pd.DataFrame(results, columns=['Person Id'] + ranked_job_titles)
    results_df["Person Id"] = results_df["Person Id"].astype(int)
    input_columns = ["Person Id", "Skillset","Computer Skills", "Work Designation", "Education", "Education Specialization", "VRF ID"]
    results_df = eval_df[input_columns].merge(results_df, on='Person Id', how='outer')
    results_df['VRF ID'] = results_df['VRF ID'].apply(lambda x: x.split('-')[1])
    
    predicted_columns = [col for col in results_df.columns if col.startswith("Predicted Rank")]
    results_df["is_in_predictions"] = results_df.apply(lambda row: row["VRF ID"] in row[predicted_columns].values, axis=1)

    accuracy = results_df["is_in_predictions"].mean()


    vrf_df = pd.read_csv(args.vrf_data_csv)
    results_df["predicted_depts"] = get_depts_from_job_df(results_df, vrf_df)
    results_dir = create_timestamped_results(args.results_dir, results_df)
    
    results_info_file = os.path.join(results_dir, args.results_info_file_name)
    results_info = f"""
    Property Graph Index Version: {latest_pg_store_dir}
    Vector Store Index Version: {latest_vector_store_dir}
    Model Accuracy: {accuracy}
    Model Precision:___
    """
    with open(results_info_file, "w") as file:
        file.write(results_info)

    logging.info("Script finished.")


if __name__ == "__main__":
    main()

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
import pandas as pd
import os

from utils import create_timestamped_results, load_cached_indexes, get_depts_from_job_df
from training_data import clean_participant_data
from pinecone_utils import get_pinecone_index


from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import Settings

from pinecone import Pinecone

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
        "--input_participant_info_csv",
        default='data/input_participant_info_raw.csv',
        help="Path to input participants information.",
    )

    parser.add_argument(
        "--input_participant_info_cleaned_csv",
        default='data/input_participant_info_cleaned.csv',
        help="Path to write cleaned participants information.",
    )

    parser.add_argument(
        "--vrf_data_cleaned_csv",
        default='data/vrf_data_cleaned.csv',
        help="Path to the vrf csv file.",
    )

    parser.add_argument(
        "--results_dir",
        default='results/',
        help="Path to model results dir.",
    )

    parser.add_argument(
        "--num_samples",
        default=20,
        type=int,
        help="Number of past participants to sample for testing.",
    )

    parser.add_argument(
        '--random_sample',
        action='store_true',
        dest='random_sample',  # Default False
        help="Enable verbose logging. (default is disabled)"
    )

    parser.add_argument(
        "--num_job_predictions",
        default=3,
        type=int,
        help="Number of jobs to predict for each participant.",
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

def create_input_str(row):
    sentences = " Participant " + str(int(row["SP ID"])) + " worked with designations: "
    for des in row["Work Experience/Designation"]:
        sentences += des + ", "
    sentences += sentences[:-2] + ". " # remove last comma
    zip_cols = [row["Education/Qualifications"], row["Education/Specialization"]]
    for qual, spec in zip(*zip_cols):
        sentences += " The participant has a " + qual + " education specialized in " + spec + ". "
    return sentences

def make_input_df(participant_info_df, num_samples, random_sample, input_columns):
    """
    Read the input participant info csv and create a dataframe with the required columns.
    
    Args:
        participant_info_df (pd.DataFrame): Path to the input participant info csv file.
        num_samples (int): Number of samples to take from the input csv.
        random_sample (bool): Whether to take a random sample or the first n samples.
    
    Returns:
        input_df (pd.DataFrame): A dataframe with the required columns
    """
    input_df = participant_info_df
    input_df = input_df.map(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)
    input_df = input_df.dropna(subset=['SP ID'])
    if random_sample:
        input_df = input_df.sample(n=num_samples)
    else:
        input_df = input_df.head(num_samples)
    input_df[input_columns] = input_df[input_columns].fillna("NA")
    input_df["input"] = input_df.apply(create_input_str, axis = 1)
    return input_df

def run_embedding_inference(input_df, vector_retriever, input_columns):
    participants_nodes = dict()
    for _, row in input_df.iterrows():
        sp_id = row["SP ID"]
        if all(row[col] == 'NA' for col in input_columns):
            participants_nodes[sp_id] = []
        else:
            summary = row["input"]
            participant_nodes = vector_retriever.retrieve(summary)
            participants_nodes[sp_id] = participant_nodes
    participant_jobs_vec_db = dict()
    for sp_id, nodes in participants_nodes.items():
        if nodes:
            job_titles = []
            request_names = []
            scores = []
            for node in nodes:
                job_titles.append(node.metadata["Job Title"])
                request_names.append(node.metadata["Request Name"])
                scores.append(node.score)
            participant_jobs_vec_db[sp_id] = (job_titles, request_names, scores)
        else:
            participant_jobs_vec_db[sp_id] = (["Ashram Support"] * 3, [""] * 3, [0] * 3)
    return participants_nodes, participant_jobs_vec_db

def jobs_dict_to_df(jobs_dict, max_jobs=3):
    """
    Convert the jobs dictionary to a dataframe.
    
    Args:
        jobs_dict (dict): A dictionary with the jobs.
    
    Returns:
        jobs_df (pd.DataFrame): A dataframe with the jobs.
    """
    rows = []
    for sp_id, jobs_tuple in jobs_dict.items():
        rows.append([sp_id] + jobs_tuple[0] + jobs_tuple[1] + jobs_tuple[2])
    
    max_columns = max(len(row) for row in rows)
    rows = [row + ["NA"] * (max_columns - len(row)) for row in rows]
    headers = ["SP ID"] + [f"Vec Pred Job Title: {i}" for i in range(1, max_jobs + 1)]
    headers += [f"Vec Pred Request Name: {i}" for i in range(1, max_jobs + 1)]
    headers += [f"Vec Pred Score: {i}" for i in range(1, max_jobs + 1)]
    jobs_df = pd.DataFrame(rows, columns=headers)
    return jobs_df



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
    embeddings = OpenAIEmbedding()
    Settings.llm = llm
    Settings.embed_model = embeddings
    
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    pinecone_vector_index = get_pinecone_index(args.pinecone_index_name, pc, create_if_not_exists=False)
    pinecone_vector_store = PineconeVectorStore(pinecone_index=pinecone_vector_index)
    vector_index = VectorStoreIndex.from_vector_store(vector_store=pinecone_vector_store)
    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=args.num_job_predictions)

    print("Peparing data for inference...")
    participant_info_df = pd.read_csv(args.input_participant_info_csv)
    input_columns = ["SP ID", "Work Experience/Designation", "Education/Qualifications", "Education/Specialization"]
    participant_info_df = participant_info_df[input_columns]
    participant_info_df = clean_participant_data(participant_info_df, target_columns=input_columns, columns_to_concatenate=["Work Experience/Designation", "Education/Qualifications", "Education/Specialization"])
    participant_info_df.to_csv(args.input_participant_info_cleaned_csv, index=False)
    input_df = make_input_df(participant_info_df, args.num_samples, args.random_sample, input_columns)
    print("Running inference on input data...")
    _, participant_jobs_vec_db = run_embedding_inference(input_df, vector_retriever, input_columns)
    vec_preds_df = jobs_dict_to_df(participant_jobs_vec_db)
    results_df = vec_preds_df.merge(input_df[input_columns], on='SP ID', how='outer')
    vrf_df = pd.read_csv(args.vrf_data_cleaned_csv)
    dept_columns = [f"Department {i}" for i in range(1, args.num_job_predictions + 1)]
    results_df[dept_columns] = get_depts_from_job_df(results_df, vrf_df, pred_column_prefix="Vec Pred Job Title", dept_columns=dept_columns)
    results_dir = create_timestamped_results(args.results_dir, results_df)
    print(f"Inference results saved to {results_dir}")
    logging.info("Script finished.")


if __name__ == "__main__":
    main()

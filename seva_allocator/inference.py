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

from utils import create_timestamped_results, extract_jobs_from_nodes, load_cached_indexes, get_depts_from_job_df
from training_data import create_vrf_single_df

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from llama_index.llms.openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
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
        "--input_participant_info_csv",
        default='data/input_participant_info.csv',
        help="Path to input participants information.",
    )

    parser.add_argument(
        "--vrf_data_csv",
        default='data/vrf_data.csv',
        help="Path to the vrf csv file.",
    )

    parser.add_argument(
        "--results_dir",
        default='results/',
        help="Path to model results dir.",
    )

    parser.add_argument(
        "--vector_store_base_dir",
        default='vector_store_versions/',
        help="Path to vector store dbs.",
    )

    parser.add_argument(
        "--vector_version",
        default='latest',
        help="The version of the index to retrieve. Default is 'latest'. Format is 'YYYYMMDD_HHMMSS'",
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

def make_input_df(input_participant_info_csv, num_samples, random_sample, input_columns):
    """
    Read the input participant info csv and create a dataframe with the required columns.
    
    Args:
        input_participant_info_csv (str): Path to the input participant info csv file.
        num_samples (int): Number of samples to take from the input csv.
        random_sample (bool): Whether to take a random sample or the first n samples.
    
    Returns:
        input_df (pd.DataFrame): A dataframe with the required columns
    """
    input_df = pd.read_csv(input_participant_info_csv)
    input_df = input_df.map(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)
    input_df = input_df.dropna(subset=['SP ID'])
    if random_sample:
        input_df = input_df.sample(n=num_samples)
    else:
        input_df = input_df.head(num_samples)
    input_df[input_columns] = input_df[input_columns].fillna("NA")
    input_df["input"] =  " Participant " + input_df["SP ID"].astype(str) + \
                            ". The participant worked with designation: " + input_df["Work Experience/Designation"] + \
                            " and has a " + input_df["Education/Qualifications"] + " education specialized in " + \
                            input_df["Education/Specialization"] + " and speaks these languages: " + input_df["Languages"]
    return input_df

def run_embedding_inference(input_df, vector_retriever, job_list, input_columns):
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
            for node in nodes:
                job_titles.append(node.metadata["Job Title"])
                request_names.append(node.metadata["Request Name"])
            participant_jobs_vec_db[sp_id] = (job_titles, request_names)
        else:
            participant_jobs_vec_db[sp_id] = (["Ashram Support"] * 3, [""] * 3)
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
        rows.append([sp_id] + jobs_tuple[0] + jobs_tuple[1])
    
    max_columns = max(len(row) for row in rows)
    rows = [row + ["NA"] * (max_columns - len(row)) for row in rows]
    headers = ["SP ID"] + [f"Vec Pred Job Title: {i}" for i in range(1, max_jobs + 1)]
    headers += [f"Vec Pred Request Name: {i}" for i in range(1, max_jobs + 1)]
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
    embeddings = OpenAIEmbeddings()
    Settings.llm = llm
    Settings.embed_model = embeddings

    _, vector_index, _, _ = load_cached_indexes(
        pg_store_base_dir="",
        vector_store_base_dir=args.vector_store_base_dir,
        pg_version="",
        vector_version=args.vector_version)
    vector_retriever = VectorIndexRetriever(
                        index=vector_index,
                        similarity_top_k=args.num_job_predictions)
    vrf_single_df = create_vrf_single_df(args.vrf_specific_train_data_csv, args.vrf_generic_train_data_csv)
    job_list = vrf_single_df["Job Title"].tolist()
    print("Peparing data for inference...")
    input_columns = ["SP ID", "Work Experience/Designation", "Education/Qualifications", "Education/Specialization", "Languages"]
    input_df = make_input_df(args.input_participant_info_csv, args.num_samples, args.random_sample, input_columns)
    print("Running inference on input data...")
    _, participant_jobs_vec_db = run_embedding_inference(input_df, vector_retriever, job_list, input_columns)
    vec_preds_df = jobs_dict_to_df(participant_jobs_vec_db)
    results_df = vec_preds_df.merge(input_df[input_columns], on='SP ID', how='outer')
    vrf_df = pd.read_csv(args.vrf_data_csv)
    dept_columns = [f"Department {i}" for i in range(1, args.num_job_predictions + 1)]
    results_df[dept_columns] = get_depts_from_job_df(results_df, vrf_df, pred_column_prefix="Vec Pred Job Title", dept_columns=dept_columns)
    results_dir = create_timestamped_results(args.results_dir, results_df)
    print(f"Inference results saved to {results_dir}")
    logging.info("Script finished.")


if __name__ == "__main__":
    main()

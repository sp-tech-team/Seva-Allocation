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


from graph_rag_lib import GraphRagRetriever, PlainQueryEngine
from utils import create_timestamped_results, extract_jobs_from_nodes, load_cached_indexes
from training_data import create_vrf_single_df

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
        "--pg_version",
        default='latest',
        help="The version of the index to retrieve. Default is 'latest'. Format is 'YYYYMMDD_HHMMSS'.",
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
        "--inference_batch_size",
        default=20,
        type=int,
        help="Number of participants to assign in one api request.",
    )

    parser.add_argument(
        "--num_job_predictions",
        default=2,
        type=int,
        help="Number of jobs to predict for each participant. Note this needs to match the prompt.",
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',  # Default False
        help="Enable verbose logging. (default is disabled)"
    )

    parser.add_argument(
        "--query_mode",
        default='PROMPT_ONLY',
        help="The mode to choose query input for the retriever. Options are 'ALL', 'PROMPT_ONLY', and 'SPECIFIC_ONLY'.",
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

    return parser.parse_args()

def make_input_df(input_participant_info_csv, num_samples, random_sample):
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
    optional_columns = ["Computer Skills", "Any Additional Skills",
                        "Work Experience/Designation", "Education/Qualifications", "Education/Specialization"]
    input_df[optional_columns] = input_df[optional_columns].fillna("NA")
    input_df["input"] =  " Participant " + input_df["SP ID"].astype(str) + \
                            ". The participant worked with designation: " + input_df["Work Experience/Designation"] + \
                            " and has a " + input_df["Education/Qualifications"] + " education specialized in " + \
                            input_df["Education/Specialization"]
    return input_df

def build_prompt(prompt_config_file):
    """
    Read the prompt config file and build the prompt string.
    
    Args:
        prompt_config_file (str): Path to the prompt config file.
    
    Returns:
        prompt (str): The prompt string.
    """
    with open(prompt_config_file, 'r') as f:
        prompt_config = json.load(f)
    prompt = ""
    if prompt_config["prompt_mode"] == "COMPLETE":
        prompt_complete_file = prompt_config["prompt_complete_file"]
        with open(prompt_complete_file, 'r') as f:
            prompt = f.read()
    return prompt

def create_specific_queries(input_df):
    """
    Create specific queries for the input dataframe.
    
    Args:
        input_df (pd.DataFrame): A dataframe with the required columns.
    
    Returns:
        specific_queries (list): A list of specific queries.
    """
    skills_query = input_df['Skills'] + ". " + \
                   input_df["Any Additional Skills"] + \
                   " and specifically computer skills: " + input_df["Computer Skills"]
    experience_query = "The participant worked with designation: " + input_df["Work Experience/Designation"]
    education_query = "The participant has a " + input_df["Education/Qualifications"] + " education specialized in " + input_df["Education/Specialization"]
    return ["\n".join(skills_query), "\n".join(experience_query), "\n".join(education_query)]

def run_embedding_inference(input_df, vector_retriever, job_list):
    participants_nodes = dict()
    for _, row in input_df.iterrows():
        summary = row["input"]
        sp_id = row["SP ID"]
        participant_nodes = vector_retriever.retrieve(summary)
        participants_nodes[sp_id] = participant_nodes
    participant_jobs_vec_db = dict()
    for sp_id, nodes in participants_nodes.items():
        jobs = extract_jobs_from_nodes(nodes, job_list)
        participant_jobs_vec_db[sp_id] = jobs
    return participants_nodes, participant_jobs_vec_db

def jobs_dict_to_df(jobs_dict):
    """
    Convert the jobs dictionary to a dataframe.
    
    Args:
        jobs_dict (dict): A dictionary with the jobs.
    
    Returns:
        jobs_df (pd.DataFrame): A dataframe with the jobs.
    """
    rows = []
    for sp_id, jobs in jobs_dict.items():
        rows.append([sp_id] + jobs)
    max_jobs = max(len(row) for row in rows)
    rows = [row + ["NA"] * (max_jobs - len(row)) for row in rows]
    headers = ["SP ID"] + [f"Vec Pred Job_{i}" for i in range(1, max_jobs)]
    jobs_df = pd.DataFrame(rows, columns=headers)
    return jobs_df


def run_inference(input_df, prompt, llm, participants_nodes, participant_jobs_vec_db, batch_size, num_job_predictions):
    """
    Run inference on the input dataframe.
    
    Args:
        input_df (pd.DataFrame): A dataframe with the required columns.
        prompt (str): The prompt string.
        query_engine (PlainQueryEngine): The query engine to use for inference.
        batch_size (int): The batch size for inference.
        num_job_predictions (int): The number of job predictions to make for each participant.
        
    Returns:
        results_df (pd.DataFrame): A dataframe with the inference results.
    """

    # Group the eval inputs into chunks of batch_size
    results = []
    for i in range(0, len(input_df), batch_size):
        chunk_df = input_df.iloc[i:i + batch_size]
        particpant_summaries = ""
        for _, row in chunk_df.iterrows():
            sp_id = row["SP ID"]
            nodes = participants_nodes[sp_id]
            jobs = participant_jobs_vec_db[sp_id]
            particpant_summaries += row["input"] + "\n"
            particpant_summaries += f"The following jobs have been assigned to participant {sp_id}: {', '.join(jobs)}\n"
            particpant_summaries += f"Here are the descriptions of the jobs above assigned to participant {sp_id}:\n" + \
                "\n".join(node.get_content() for node in nodes) + "\n\n"
            
        prompt_with_summaries = prompt + particpant_summaries
        response = llm.complete(prompt_with_summaries)
        print("Model Response String: \n", response)
        lines = response.text.splitlines()
        for line in lines:
            predictions = line.split(",")
            results.append(predictions[:5])
    
    max_length = max(len(row) for row in results)
    results = [row + ["NA"] * (max_length - len(row)) for row in results]
    column_names = ["SP ID", "Rating"] + [f"Predicted Rank {i} Job Title" for i in range(1, num_job_predictions + 1)]
    column_names.append("Clusters")
    #extra_columns = [f"extra_{i}" for i in range(1, max_length - num_job_predictions)]
    #column_names += extra_columns
    results_df = pd.DataFrame(results, columns=column_names)
    results_df["SP ID"] = results_df["SP ID"].astype(int)
    return results_df

def get_depts_from_job(job_title, vrf_df):
    return vrf_df[vrf_df['Job Title'] == job_title]['Department'].drop_duplicates().values

def get_depts_from_job_df(results_df, vrf_df):
    """
    Get the departments for the predicted jobs in the results dataframe.
    
    Args:
        results_df (pd.DataFrame): The results dataframe.
        vrf_df (pd.DataFrame): The vrf dataframe.
    
    Returns:
        depts (pd.Series): A series with the departments for the predicted jobs.
    """
    predicted_columns = [col for col in results_df.columns if col.startswith("Predicted Rank")]
    def get_depts(row):
        depts = ""
        for job_title in row[predicted_columns]:
            if pd.isna(job_title):
                depts += ",NA, "
            else:
                job_depts = get_depts_from_job(job_title, vrf_df)
                depts += job_title + ": " + ", ".join(job_depts) + " "
        return depts
    depts = results_df.apply(get_depts, axis=1)
    return depts

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

    pg_index, vector_index, _, _ = load_cached_indexes(
        pg_store_base_dir=args.pg_store_dir,
        vector_store_base_dir=args.vector_store_dir,
        pg_version=args.pg_version,
        vector_version=args.vector_version)
    pg_retriever = pg_index.as_retriever(include_text=False)
    vector_retriever = VectorIndexRetriever(
                        index=vector_index,
                        similarity_top_k=3)

    print("Making Graph RAG retriever...")
    graph_rag_retriever = GraphRagRetriever(vector_retriever, pg_retriever, db_mode="VECTOR_ONLY")

    # create response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
    )
    print("Creating query engines...")
    vrf_single_df = create_vrf_single_df(args.vrf_specific_train_data_csv, args.vrf_generic_train_data_csv)
    job_list = vrf_single_df["Job Title"].tolist()
    # graph_rag_query_engine = RetrieverQueryEngine(
    #     retriever=graph_rag_retriever,
    #     response_synthesizer=response_synthesizer,
    # )
    # vector_query_engine = vector_index.as_query_engine()
    # pg_keyword_query_engine = pg_index.as_query_engine(
    #     include_text=False,
    #     retriever_mode="keyword",
    #     response_mode="tree_summarize",
    # )
    print("Peparing data for inference...")
    input_df = make_input_df(args.input_participant_info_csv, args.num_samples, args.random_sample)
    print("Preparing prompt...")
    prompt = build_prompt(args.prompt_config_json)
    print("Running inference on input data...")
    participants_nodes, participant_jobs_vec_db = run_embedding_inference(input_df, graph_rag_retriever, job_list)
    # results_df = run_inference(input_df, prompt,
    #                            llm,
    #                            participants_nodes,
    #                            participant_jobs_vec_db,
    #                            args.inference_batch_size,
    #                            args.num_job_predictions)
    vec_preds_df = jobs_dict_to_df(participant_jobs_vec_db)
    #results_df = vec_preds_df.merge(results_df, on='SP ID', how='outer')
    # input_columns = ["SP ID", "Skills", "Any Additional Skills", "Computer Skills", "Work Experience/Designation", "Education/Qualifications", "Education/Specialization"]
    input_columns = ["SP ID", "Work Experience/Designation", "Education/Qualifications", "Education/Specialization"]
    results_df = vec_preds_df.merge(input_df[input_columns], on='SP ID', how='outer')
    # vrf_df = pd.read_csv(args.vrf_data_csv)
    #results_df["predicted_depts"] = get_depts_from_job_df(results_df, vrf_df)
    results_dir = create_timestamped_results(args.results_dir, results_df)
    print(f"Inference results saved to {results_dir}")
    logging.info("Script finished.")


if __name__ == "__main__":
    main()

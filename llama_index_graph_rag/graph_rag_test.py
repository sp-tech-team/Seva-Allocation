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


from graph_rag_lib import GraphRagRetriever, CustomQueryEngine
from utils import create_timestamped_results, load_cached_indexes
from training_data import create_vrf_single_df
from graph_rag_inference import run_inference, build_prompt, get_depts_from_job_df

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


def test_parse_args() -> argparse.Namespace:
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
        default='eval_results/',
        help="Path to model eval results dir.",
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
        default='ALL',
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

def make_eval_input_df(past_participant_info_csv, num_samples, random_sample):
    """
    Create a dataframe of past participants to evaluate the model on.
    
    Args:
        past_participant_info_csv: str, path to the past participants csv file.
        num_samples: int, number of samples to take from the past participants.
        random_sample: bool, whether to take a random sample or the first n samples.
    
    Returns:
        eval_input_df, a dataframe of past participants to evaluate the model on.        
    """
    eval_input_df = pd.read_csv(past_participant_info_csv)
    eval_input_df = eval_input_df.map(lambda x: x.replace("\n", " ") if isinstance(x, str) else x)
    eval_input_df = eval_input_df[eval_input_df['Seva Allocation Accurate or not']==1]
    eval_input_df= eval_input_df.dropna(subset=['Person Id', 'Skillset', 'VRF ID'])
    if random_sample:
        eval_input_df = eval_input_df.sample(n=num_samples)
    else:
        eval_input_df = eval_input_df.head(num_samples)
    optional_columns = ["Computer Skills", "Additional Skills", "Work Designation", "Education", "Education Specialization"]
    eval_input_df[optional_columns] = eval_input_df[optional_columns].fillna("NA")
    eval_input_df["input"] =  " Participant " + eval_input_df["Person Id"].astype(str) + " has skills: " + eval_input_df['Skillset'] + \
                            ". " + eval_input_df["Additional Skills"] + " and specifically computer skills: " + eval_input_df["Computer Skills"] + \
                            ". The participant worked with designation: " + eval_input_df["Work Designation"] + \
                            " and has a " + eval_input_df["Education"] + " education specialized in " + eval_input_df["Education Specialization"]
    return eval_input_df

def eval_results(results_df, eval_input_df, vrf_df):
    """
    Evaluate the results of the model on the evaluation data.
    
    Args:
        results_df: pd.DataFrame, the results of the model on the evaluation data.
        eval_input_df: pd.DataFrame, the evaluation data.
        vrf_df: pd.DataFrame, the vrf data.
    """
    input_columns = ["Person Id", "Skillset","Computer Skills", "Work Designation", "Education", "Education Specialization", "VRF ID"]
    results_df = eval_input_df[input_columns].merge(results_df, on='Person Id', how='outer')
    results_df['VRF ID'] = results_df['VRF ID'].apply(lambda x: x.split('-')[1])
    predicted_columns = [col for col in results_df.columns if col.startswith("Predicted Rank")]
    results_df["is_in_predictions"] = results_df.apply(lambda row: row["VRF ID"] in row[predicted_columns].values, axis=1)
    accuracy = results_df["is_in_predictions"].mean()
    
    results_df["predicted_depts"] = get_depts_from_job_df(results_df, vrf_df)
    return results_df, accuracy

def main() -> None:
    """Main entry point of the script."""
    args = test_parse_args()

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

    pg_index, vector_index, pg_store_dir, vector_store_dir = load_cached_indexes(
        pg_store_base_dir=args.pg_store_dir,
        vector_store_base_dir=args.vector_store_dir,
        pg_version=args.pg_version,
        vector_version=args.vector_version)
    pg_retriever = pg_index.as_retriever(include_text=False)
    vector_retriever = VectorIndexRetriever(index=vector_index)

    print("Making Graph RAG retriever...")
    graph_rag_retriever = GraphRagRetriever(vector_retriever, pg_retriever)

    # create response synthesizer
    response_synthesizer = get_response_synthesizer(
        response_mode="tree_summarize",
    )
    print("Creating query engines...")
    vrf_single_df = create_vrf_single_df(args.vrf_specific_train_data_csv, args.vrf_generic_train_data_csv)
    jobs_list = vrf_single_df["Job Title"].tolist()
    graph_rag_query_engine = CustomQueryEngine(
        retriever=graph_rag_retriever,
        response_synthesizer=response_synthesizer,
        query_mode=args.query_mode,
        job_list=jobs_list,
    )
    vector_query_engine = vector_index.as_query_engine()
    pg_keyword_query_engine = pg_index.as_query_engine(
        # setting to false uses the raw triplets instead of adding the text from the corresponding nodes
        include_text=False,
        retriever_mode="keyword",
        response_mode="tree_summarize",
    )
    print("Peparing data for inference / evaluation...")
    eval_input_df = make_eval_input_df(args.past_participant_info_csv, args.num_samples, args.random_sample)
    print("Preparing prompt...")
    prompt = build_prompt(args.prompt_config_json)
    print("Running inference on evaluation data...")
    results_df = run_inference(eval_input_df, prompt, graph_rag_query_engine, args.inference_batch_size, args.num_job_predictions)
    print("Evaluating the results...")
    vrf_df = pd.read_csv(args.vrf_data_csv)
    results_df, accuracy = eval_results(results_df, eval_input_df, vrf_df)
    print("Saving evaluation results...")
    results_dir = create_timestamped_results(args.results_dir, results_df)
    results_info_file = os.path.join(results_dir, args.results_info_file_name)
    results_info = f"""
Property Graph Index Version: {pg_store_dir}
Vector Store Index Version: {vector_store_dir}
Model Accuracy: {accuracy}
Model Precision:___
Prompt Used: \n{prompt}
    """
    with open(results_info_file, "w") as file:
        file.write(results_info)
    logging.info("Script finished.")


if __name__ == "__main__":
    main()

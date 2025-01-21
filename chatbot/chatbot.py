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
import os
import pandas as pd
from io import StringIO

# from preprocessing.participant_data import  create_participant_db_df

from openai import OpenAI

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
        "-participant_pinecone_index_name",
        default='participant-test-local',
        help="Path to vector store dbs.",
    )

    parser.add_argument(
        "-vrf_pinecone_index_name",
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


class CombinedRetriever:
    def __init__(self, retrievers):
        self.retrievers = retrievers

    def retrieve(self, query):
        # Aggregate results from both retrievers
        results = []
        for retriever in self.retrievers:
            results.extend(retriever.retrieve(query))
        return results

prompt_tmpl = \
"""
{query_str}\n
Here is the context... \n
{context_str}
"""

def initialize_pipeline(input_participant_info_csv):
    client = OpenAI()
    # Load data
    # target_columns = ['SP ID', 'Work Experience/Company', 'Work Experience/Designation',
    #    'Work Experience/Tasks', 'Work Experience/Industry',
    #    'Education/Qualifications', 'Education/Specialization',
    #    'Any Additional Skills', 'Computer Skills', 'Skills',
    #    'Languages', 'Gender', 'Age', 'Work Experience/From Date', 'Work Experience/To Date']
    # participant_info_raw_df = pd.read_csv('data/input_participant_info_raw.csv')
    # participant_db_df = create_participant_db_df(participant_info_raw_df, target_columns)
    participant_db_df = pd.read_csv("participant_test.csv")
    csv_buffer = StringIO()
    participant_db_df.to_csv(csv_buffer, index=False)
    context = csv_buffer.getvalue()
    return context, client

def process_input(user_input, context, client):
    query = prompt_tmpl.format(context_str=context, query_str=user_input)
    completion = client.chat.completions.create(
        model="gpt-4o", #"o1-mini",
        messages=[
            #{"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": query
            }
        ]
    )
    response = completion.choices[0].message.content

    print(f"LlamaIndex: \n{response}")

def main() -> None:
    """Main entry point of the script."""
    args = inference_parse_args()

    # Configure logging
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    if not args.verbose:
        logging.disable(logging.CRITICAL)

    logging.info("Script started.")

    context, client = initialize_pipeline(args.input_participant_info_csv)
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        try:
            process_input(user_input, context, client)
        except Exception as e:
            print(f"An error occurred: {e}")

    logging.info("Script finished.")


if __name__ == "__main__":
    main()

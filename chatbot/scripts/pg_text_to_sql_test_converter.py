import pdb
import os
import ast
import pandas as pd
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--test_file_json',
        type=str,
        help='Test file in JSON format',
        default=None
    )

    group.add_argument(
        '--test_file_csv',
        type=str,
        help='Test file in CSV format',
        default=None
    )

    parser.add_argument(
        '--output_file_json',
        type=str,
        help='File to save the output JSON',
        default=None
    )

    parser.add_argument(
        '--output_file_csv',
        type=str,
        help='File to save the output csv',
        default=None
    )

    return parser.parse_args()

def convert_df_to_json_format(df):
    result = {}
    # Group by "Test Name"
    for test_name, group in df.groupby("Test Name"):
        # Assume the same database for all rows in this test group
        database_name = group["Database Name"].iloc[0]

        questions = []
        for _, row in group.iterrows():
            question = {
                "question_id": str(row["Question ID"]),
                "question": row["Question"],
                "answer": ast.literal_eval(row["Answer"]),  # parse string to list
                "notes": row["Notes"]
            }
            questions.append(question)

        result[test_name] = {
            "database_name": database_name,
            "questions": questions
        }
    return result


def convert_json_to_df(test_json):
    records = []
    for test_name, test_data in test_json.items():
        database_name = test_data["database_name"]
        for question in test_data["questions"]:
            record = {
                "Test Name": test_name,
                "Database Name": database_name,
                "Question ID": question["question_id"],
                "Question": question["question"],
                "Answer": str(question["answer"]),  # convert list to string
                "Notes": question["notes"]
            }
            records.append(record)
    return pd.DataFrame(records)

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    if bool(args.test_file_json) == bool(args.test_file_csv):
        raise ValueError("Only one of --test_file_json or --test_file_csv must be provided.")
    if bool(args.output_file_json) == bool(args.output_file_csv):
        raise ValueError("Only one of --output_file_json or --output_file_csv must be provided.")
    if bool(args.test_file_csv) and not bool(args.output_file_json):
        raise ValueError("If --test_file_csv is provided, --output_file_json must also be provided.")
    if bool(args.test_file_json) and not bool(args.output_file_csv):
        raise ValueError("If --test_file_json is provided, --output_file_csv must also be provided.")
    if args.test_file_csv and args.test_file_csv.endswith(".csv") == False:
        raise ValueError("The --test_file_csv must be a .csv file.")
    if args.test_file_json and args.test_file_json.endswith(".json") == False:
        raise ValueError("The --test_file_json must be a .json file.")
    if args.output_file_csv and args.output_file_csv.endswith(".csv") == False:
        raise ValueError("The --output_file_csv must be a .csv file.")
    if args.output_file_json and args.output_file_json.endswith(".json") == False:
        raise ValueError("The --output_file_json must be a .json file.")

    if args.test_file_csv:
        df = pd.read_csv(args.test_file_csv)
        result = convert_df_to_json_format(df)
        with open(args.output_file_json, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Converted CSV to JSON and saved to {args.output_file_json}")
    elif args.test_file_json:
        with open(args.test_file_json, "r") as f:
            test_json = json.load(f)
        # Convert the JSON data to a DataFrame
        df = convert_json_to_df(test_json)
        df.to_csv(args.output_file_csv, index=False)
        print(f"Converted JSON to CSV and saved to {args.output_file_csv}")
    else:
        raise ValueError("Please provide a valid test file in either JSON or CSV format.")

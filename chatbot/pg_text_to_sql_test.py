import pdb
import os
import json
import argparse
import pandas as pd
from chatbot.pg_text_to_sql import Text2PGSQL
from database.participant_pg_database import load_participant_db, DbConfig, pretty_print_sqlalchemy_results

from dotenv import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test_file_json',
        type=str,
        help='JSON file containing test queries',
        default="chatbot/test_data/tests2_converted.json"
    )

    return parser.parse_args()

query_output_formatter_str = """
Note for the follow query that will be asked below:
SPID should be the only finally selected column when generating SQL code...

"""

def evaluate_results(expected_results, actual_df, result_column="SP ID", include_samples=True):
    """
    Compares expected results (list of strings) with actual results in a DataFrame.
    Returns the counts and calculated metrics for a single test case.
    """
    expected_set = set(expected_results)
    actual_set = set(actual_df[result_column].tolist())

    hits = list(expected_set.intersection(actual_set)) # True Positives
    missing = list(expected_set - actual_set) # False Negatives
    extra = list(actual_set - expected_set) # False Positives

    
    # True Positives for this test case: elements that are correctly retrieved.
    tp = len(hits)
    # False Negatives: expected but missing.
    fn = len(missing)
    # False Positives: items returned that shouldn't be.
    fp = len(extra)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    samples = dict()
    if include_samples:
        samples = {
                "hits (TPs)": ",".join(hits),
                "missing (FNs)": ",".join(missing),
                "extra (FPs)": ",".join(extra)
            }
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "samples": samples
    }

def answer_question(text_to_sql, question):
    text_query = query_output_formatter_str + question["question"]
    # Generate the SQL query with vector search placeholders and the corresponding search texts.
    sql_output = text_to_sql.write_query(text_query)
    annotated_query = sql_output["query"]
    final_query = sql_output["query"]
    error_str = "None"
    if "vector_searches" in sql_output:
        # Inject embeddings into the SQL query by replacing the placeholders.
        annotated_query = text_to_sql.inject_raw_search_strings(sql_output["query"], sql_output["vector_searches"])
        final_query = text_to_sql.inject_embeddings(sql_output["query"], sql_output["vector_searches"])    
    # Execute the SQL query.
    results_df = pd.DataFrame({'SP ID': []})
    sql_results = text_to_sql.execute_query(final_query)
    crashed = False
    if sql_results["success"]:
        results_df = pd.DataFrame(sql_results['results'])
    else:
        print("SQL Query Execution Failed for question: ", question["question"])
        crashed = True
        error_str = sql_results["error"]

    return results_df, crashed, annotated_query, error_str

def run_eval_test(text_to_sql, test_queries_cfg, eval_results):
    # For macro averaging: collect each test case's metrics.
    macro_precisions = []
    macro_recalls = []
    macro_f1_scores = []

    # For micro averaging: sum TP, FP, and FN over all test cases.
    total_tp = total_fp = total_fn = 0
    # Evaluate each question in the current test
    basic_mock_tests = test_queries_cfg[test_name]["questions"]
    for idx, question in enumerate(basic_mock_tests):
        results_df, crashed, annotated_query, error_str = answer_question(text_to_sql, question)
        expected_results = question["answer"]
        if "SP ID" not in results_df.columns:
            print(f"Warning: 'SP ID' column not found in results for question {question['question_id']}.")
            results_df["SP ID"] = []
        eval_dict = evaluate_results(expected_results, results_df)
        eval_results[test_name]["local_evals"].append({
            "question_id": question["question_id"],
            "question": question["question"],
            "crashed": crashed,
            "eval": eval_dict,
            "debug_query": annotated_query,
            "error_str": error_str
        })
        # Macro: Append metrics for later averaging.
        macro_precisions.append(eval_dict["precision"])
        macro_recalls.append(eval_dict["recall"])
        macro_f1_scores.append(eval_dict["f1"])

        # Micro: Sum the counts.
        total_tp += eval_dict["tp"]
        total_fp += eval_dict["fp"]
        total_fn += eval_dict["fn"]

    # Compute micro-averaged metrics by summing over test cases.
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    eval_results[test_name]["micro_avg"] = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1": micro_f1
    }

    # Compute macro-averaged metrics (arithmetic mean of per-test-case scores).
    macro_precision = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 1.0
    macro_recall = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 1.0
    macro_f1 = sum(macro_f1_scores) / len(macro_f1_scores) if macro_f1_scores else 1.0
    eval_results[test_name]["macro_avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }

def make_eval_results_tables(eval_results):
    local_records = []
    global_records = []
    for test_name, suite_data in eval_results.items():
        # ========= 1. Local Evaluation Table =========
        for question in suite_data.get("local_evals", []):
            row = {
                "test_name": test_name,
                "question_id": question["question_id"],
                "question": question["question"],
                "crashed": question["crashed"],
                "debug_query": question["debug_query"],
                "tp": question["eval"]["tp"],
                "fn": question["eval"]["fn"],
                "fp": question["eval"]["fp"],
                "precision": question["eval"]["precision"],
                "recall": question["eval"]["recall"],
                "f1": question["eval"]["f1"],
                "hits (TPs)": question["eval"]["samples"]["hits (TPs)"],
                "missing (FNs)": question["eval"]["samples"]["missing (FNs)"],
                "extra (FPs)": question["eval"]["samples"]["extra (FPs)"],
                "error_str": question["error_str"]
            }
            local_records.append(row)
        
        # ========= 2. Global Evaluations Table =========
        for avg_type in ["micro_avg", "macro_avg"]:
            avg_data = suite_data.get(avg_type, {})
            row = {
                "test_name": test_name,
                "average_type": avg_type.replace("_avg", ""),  # -> micro / macro
                "precision": avg_data.get("precision"),
                "recall": avg_data.get("recall"),
                "f1": avg_data.get("f1")
            }
            global_records.append(row)

    local_evals_df = pd.DataFrame(local_records)
    global_evals_df = pd.DataFrame(global_records)

    return local_evals_df, global_evals_df

if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    
    args = parse_args()
    
    with open(args.test_file_json, "r") as file:
        test_queries_cfg = json.load(file)
    
    eval_results = dict()

    for test_name in test_queries_cfg.keys():
        eval_results[test_name] = {
            "database_name": test_queries_cfg[test_name]["database_name"],
            "local_evals": [],
            "micro_avg": {},
            "macro_avg": {}
        }
        
        db_config = DbConfig(
            os.getenv("DB_USER"),
            os.getenv("DB_HOST"),
            os.getenv("DB_PORT"),
            test_queries_cfg[test_name]["database_name"],
            os.getenv("DB_PASSWORD"),
            os.getenv("OPENAI_API_KEY")
        )
        participant_db = load_participant_db(db_config)
        text_to_sql = Text2PGSQL(participant_db)
        run_eval_test(text_to_sql, test_queries_cfg, eval_results)

    eval_results_json = json.dumps(eval_results, indent=4)
    print(eval_results_json)
    with open("chatbot/test_results/eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)
    local_evals_df, global_evals_df = make_eval_results_tables(eval_results)
    local_evals_df.to_csv("chatbot/test_results/local_evals_results2.csv", index=False)
    global_evals_df.to_csv("chatbot/test_results/global_evals_results2.csv", index=False)
    print("\n=== Evaluation Results Saved ===")

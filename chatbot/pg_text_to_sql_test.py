import pdb
import os
import json
import yaml
import argparse
import pandas as pd
from chatbot.pg_text_to_sql import Text2PGSQL
from database.participant_pg_database import load_participant_db, DbConfig
from chatbot.scripts.pg_text_to_sql_test_converter import convert_df_to_json_format


from dotenv import load_dotenv
load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--test_file_json',
        type=str,
        help='JSON file containing test queries',
    )
    parser.add_argument(
        '--test_file_csv',
        type=str,
        help='csv file containing test queries',
    )
    parser.add_argument(
        "--run_tests_filter",
        nargs="*",
        help=f"List of test names to run. example: --run_tests_filter 'Basic Question Answer Pairs - Mock 1' 'Basic Question Answer Pairs - Mock 2'",
    )
    parser.add_argument(
        "--results_file_json",
        type=str,
        default="chatbot/test_results/eval_results.json",
        help="Path to save the evaluation results JSON file"
    )
    parser.add_argument(
        "--prompts_config_yaml",
        type=str,
        default="chatbot/configs/text_to_sql_prompts.yaml",
        help="Path to the YAML file containing the prompt configuration"
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="prompt_pg_vector_1",
        help="Key for the prompt to use from the YAML file"
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

def run_eval_test(text_to_sql, test_queries_cfg, eval_results, test_name):
    # For macro averaging: collect each test case's metrics.
    macro_precisions = []
    macro_recalls = []
    macro_f1_scores = []

    # For micro averaging: sum TP, FP, and FN over all test cases.
    total_tp = total_fp = total_fn = 0
    # Evaluate each question in the current test
    test = test_queries_cfg[test_name]["questions"]
    for idx, question in enumerate(test):
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

    if args.test_file_json is None and args.test_file_csv is None:
        raise ValueError("Please provide either a JSON or CSV file containing test queries.")
    if args.test_file_json  and args.test_file_csv:
        raise ValueError("Please provide only one of the JSON or CSV files containing test queries.")
    
    if args.test_file_json:
        with open(args.test_file_json, "r") as file:
            test_queries_cfg = json.load(file)
    elif args.test_file_csv:
        test_df = pd.read_csv(args.test_file_csv)
        test_queries_cfg = convert_df_to_json_format(test_df)
    # Filter the test queries based on the provided filter
    if args.run_tests_filter:
        test_queries_cfg = {
            name: cfg for name, cfg in test_queries_cfg.items()
            if name in args.run_tests_filter
        }
    
    db_config = DbConfig(
        os.getenv("SUPABASE_USER"),
        os.getenv("SUPABASE_HOST"),
        os.getenv("SUPABASE_PORT"),
        os.getenv("SUPABASE_NAME"),
        os.getenv("SUPABASE_PASSWORD")
    )

    participant_db = load_participant_db(db_config, 'participants')
    with open(args.prompts_config_yaml, "r") as f:
        prompt_tmpls = yaml.safe_load(f)
    text_to_sql_prompt_tmpl = prompt_tmpls[args.prompt_key]
    eval_results = dict()
    for test_name in test_queries_cfg.keys():
        table_base_name = test_queries_cfg[test_name]["table_base_name"]
        eval_results[test_name] = {
            "table_base_name": table_base_name,
            "local_evals": [],
            "micro_avg": {},
            "macro_avg": {}
        }
        participant_db.reset_table_base_name(table_base_name)
        text_to_sql = Text2PGSQL(participant_db, text_to_sql_prompt_tmpl)
        run_eval_test(text_to_sql, test_queries_cfg, eval_results, test_name)

    eval_results_json = json.dumps(eval_results, indent=4)
    print(eval_results_json)
    os.makedirs(os.path.dirname(args.results_file_json), exist_ok=True)
    with open(args.results_file_json, "w") as f:
        json.dump(eval_results, f, indent=4)
    local_evals_df, global_evals_df = make_eval_results_tables(eval_results)
    local_evals_df.to_csv("chatbot/test_results/local_evals_results.csv", index=False)
    global_evals_df.to_csv("chatbot/test_results/global_evals_results.csv", index=False)
    print("\n=== Evaluation Results Saved ===")

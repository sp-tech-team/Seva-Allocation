import pdb
import argparse
import json
from typing_extensions import Annotated, TypedDict
import ast
import signal
import os
import pandas as pd

from langchain import hub
from langchain_core.documents.base import Document
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from chatbot.pg_text_to_sql import Text2PGSQL
from database.participant_pg_database import DbConfig, load_participant_db, format_query_result

from dotenv import load_dotenv
load_dotenv()

# Default configuration
def DEFAULT_CONFIG():
    return {
        "use_vector_search": True,
        "vector_search_k": 10,
        "use_llm_for_column_selection": False,
        "use_mock_data": False
    }

# Parse configuration from command-line flags
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--config_file_json',
        type=str,
        help='Path to the JSON config file',
        default=None)
    parser.add_argument(
        '--table_base_name',
        type=str,
        help='Name of the table to query',
        default=''
    )

    return parser.parse_args()


process_text_to_sql_prompt_tmpl = """\
A text to sql processes has been run on the following question:
{input_text}

The generated SQL query was executed on the database and returned the following results:
{pretty_sql_results}

Please refine these results only if needed to answer the question in any way the text to 
sql process did not do.
"""

class ChatbotPipeline:
    def __init__(self, config, text_to_sql):
        self.config = config
        self.text_to_sql = text_to_sql
        self.llm = ChatOpenAI(model="gpt-4o")
        self.embedding_model = OpenAIEmbeddings()
    

    def chatbot(self, text_query):

        sql_output = self.text_to_sql.write_query(text_query)
        annotated_query = sql_output["query"]
        final_query = sql_output["query"]
        error_str = "None"
        if "vector_searches" in sql_output:
            # Inject embeddings into the SQL query by replacing the placeholders.
            annotated_query = self.text_to_sql.inject_raw_search_strings(sql_output["query"], sql_output["vector_searches"])
            final_query = self.text_to_sql.inject_embeddings(sql_output["query"], sql_output["vector_searches"])    
        # Execute the SQL query.
        results_df = pd.DataFrame({'SP ID': []})
        sql_results = self.text_to_sql.execute_query(final_query)
        crashed = False
        final_response = ""
        pretty_sql_results = ""
        if sql_results["success"]:
            results_df = pd.DataFrame(sql_results['results'])
            pretty_sql_results = results_df.to_markdown(index=False)
            process_text_to_sql_prompt = process_text_to_sql_prompt_tmpl.format(
                    input_text=text_query,
                    pretty_sql_results=pretty_sql_results
                )
            print("Prompt to LLM: ", process_text_to_sql_prompt)
            final_response = self.llm.invoke(process_text_to_sql_prompt)
        else:
            crashed = True
            error_str = sql_results["error"]
            final_response = f"SQL Query Execution Failed for question: {text_query}. Error: {error_str}"

        return final_response, pretty_sql_results

def handle_exit(signal_received, frame):
    print("\n[INFO] Chatbot exiting... Conversation saved.")
    exit(0)  # Ensures clean exit

# Register signal handler for `Ctrl+C`
signal.signal(signal.SIGINT, handle_exit)

if __name__ == "__main__":

    args = parse_args()
    if args.config_file_json:
        with open(args.config_file_json, 'r') as config_file:
            config = json.load(config_file)
    else:
        config = DEFAULT_CONFIG()
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    db_config = DbConfig(
        os.getenv("SUPABASE_USER"),
        os.getenv("SUPABASE_HOST"),
        os.getenv("SUPABASE_PORT"),
        os.getenv("SUPABASE_NAME"),
        os.getenv("SUPABASE_PASSWORD"),
        os.getenv("OPENAI_API_KEY")
    )

    participant_db = load_participant_db(db_config, args.table_base_name)
    text_to_sql = Text2PGSQL(participant_db)
    pipeline = ChatbotPipeline(config, text_to_sql)
    print("Chatbot is running. Type your query below (or type 'exit' to quit):")
    while True:
        print("\n")
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        response, pretty_sql_results = pipeline.chatbot(user_input)
        log_file = "chatbot/chatbot_conversation_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"You: \n{user_input}\n\n")
            f.write(f"Bot: \n{response}\n\n\n")
        print(f"Chatbot: {response}")
        print(f"SQL Results: \n{pretty_sql_results}\n")
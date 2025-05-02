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
from database.participant_pg_database import DbConfig, load_participant_db

from dotenv import load_dotenv
load_dotenv()

# Parse configuration from command-line flags
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--table_base_name',
        type=str,
        help='Name of the table to query',
        default='participants'
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
    def __init__(self, text_to_sql):
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
        final_response = ""
        pretty_sql_results = ""
        if sql_results["success"]:
            results_df = pd.DataFrame(sql_results['results'])
            pretty_sql_results = results_df.to_markdown(index=False)
        else:
            error_str = sql_results["error"]
            final_response = f"SQL Query Execution Failed for question: {text_query}. Error: {error_str}"

        return pretty_sql_results

def handle_exit(signal_received, frame):
    print("\n[INFO] Chatbot exiting... Conversation saved.")
    exit(0)  # Ensures clean exit

# Register signal handler for `Ctrl+C`
signal.signal(signal.SIGINT, handle_exit)

if __name__ == "__main__":

    args = parse_args()

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
    pipeline = ChatbotPipeline(text_to_sql)
    print("Chatbot is running. Type your query below (or type 'exit' to quit):")
    while True:
        print("\n")
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chatbot. Goodbye!")
            break
        pretty_sql_results = pipeline.chatbot(user_input)
        log_file = "chatbot/chatbot_conversation_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"You: \n{user_input}\n\n")
            f.write(f"Bot: \n{pretty_sql_results}\n\n\n")
        print(f"SQL Results: \n{pretty_sql_results}\n")
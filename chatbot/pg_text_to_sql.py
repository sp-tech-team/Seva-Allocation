import pdb
import yaml
import os
import argparse
import re
from typing_extensions import Annotated, TypedDict
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain import hub
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from database.participant_pg_database import DbConfig, load_participant_db, pretty_print_sqlalchemy_results
from dotenv import load_dotenv
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--table_base_name',
        type=str,
        help='Name of the table to query',
        default='participants'
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


class SQLGenOutput(TypedDict):
    """Generated SQL query output including vector search mappings."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]
    vector_searches: Annotated[
        dict[str, str],
        ...,
        "Mapping from placeholder names (e.g., 'query_vector_1') to search query texts that should be embedded."
    ]

text_to_sql_tmpl = """\
Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
You can order the results by a relevant column to return the most interesting examples in the database.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist.
Pay attention to which column is in which table. Also, qualify column names with the table name when needed.

IMPORTANT NOTE: You can use specialized pgvector syntax (`<=>`) to do nearest neighbors/semantic search to a given vector from an embeddings column in the table.
The embeddings value for a given row typically represents the semantic meaning of that row.
**Do NOT fill in the vector values directly**, but rather specify placeholders in the format `[query_vector_X]` where X is a number.
In addition, provide a JSON object mapping each placeholder (e.g., "query_vector_1") to the search string that should be embedded.
For instance, if the user question is "Show me participants with experience in data science and machine learning" and your query uses two vector search placeholders, then your output should include:

"query": "SELECT ... WHERE field1_embedding <=> '[query_vector_1]' < 0.2 ... AND field2_embedding <=> '[query_vector_2]' < 0.2 ...",
"vector_searches": 
    "query_vector_1": "data science",
    "query_vector_2": "machine learning"

Also note that you should treat From and To Dates as arrays of strings not Dates. When no date is present "NA" will fill one entry of the array.
Only use tables listed below.
{schema}

Question: {input}
SQLQuery:\
"""

def strip_vector_values(sql: str) -> str:
    """
    Replaces any vector-like substrings (e.g., [0.123, -0.456, ...]) in a SQL string with '[...]'.
    This is useful for debugging or printing queries without long vector values.
    
    Args:
        sql (str): The SQL query string.
    
    Returns:
        str: The SQL query with vector values replaced by '[...]'.
    """
    # Match square-bracketed lists with floats or integers: [0.12, -1.23, ...]
    vector_pattern = re.compile(r"\[(?:\s*-?\d+(?:\.\d+)?\s*,?)*\]")

    # Replace all matches with '[...]'
    return vector_pattern.sub("[...]", sql)

class Text2PGSQL:
    def __init__(self, participant_db, text_to_sql_prompt_tmpl: str):
        self.participant_db = participant_db
        self.llm = ChatOpenAI(model="gpt-4o")
        
        self.structured_cols = self.participant_db.get_structured_table_column_names()
        self.unstructured_cols = self.participant_db.get_unstructured_table_column_names()
        self.query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
        self.vector_sql_prompt_template = ChatPromptTemplate.from_template(text_to_sql_prompt_tmpl)
        self.embedding_model = OpenAIEmbeddings()

    def write_query(self, text_query: str) -> SQLGenOutput:
        """Generate SQL query to fetch information and extract vector search strings."""
        prompt = self.vector_sql_prompt_template.invoke(
            {
                "dialect": self.participant_db.lc_db.dialect,
                "schema": self.participant_db.lc_db.get_table_info(table_names=self.participant_db.get_table_names_limited()),
                "input": text_query,
            }
        )
        structured_llm = self.llm.with_structured_output(SQLGenOutput)
        result = structured_llm.invoke(prompt)
        return result

    def inject_embeddings(self, sql_query: str, vector_searches: dict[str, str]) -> str:
        """Replace all vector search placeholders with actual embeddings.
        
        For each key in the `vector_searches` mapping (like "query_vector_1"), 
        compute the embedding for the corresponding search text and then format it as a PostgreSQL array literal.
        Replace the placeholder (e.g., "[query_vector_1]") in the SQL query with that literal.
        """
        final_query = sql_query
        for placeholder, search_text in vector_searches.items():
            # Compute the embedding vector for the search text.
            embedding_vector = self.embedding_model.embed_query(search_text)
            # Format the embedding vector as a PostgreSQL array literal, e.g., [0.123456,0.234567,...]
            formatted_embedding = f"[{', '.join(map(str, embedding_vector))}]"
            # Replace the placeholder in the SQL query. The placeholder in the query is expected to be enclosed in square brackets.
            final_query = final_query.replace(f"[{placeholder}]", formatted_embedding)
        return final_query
    
    def inject_raw_search_strings(self, sql_query: str, vector_searches: dict[str, str]) -> str:
        """
        Replace all vector search placeholders with the raw search string for logging purposes.
        
        For each key in the `vector_searches` mapping (like "query_vector_1"),
        take the corresponding search text and replace the placeholder (e.g., "[query_vector_1]")
        in the SQL query with that search text wrapped in quotes.
        
        Args:
            sql_query (str): The SQL query with placeholders.
            vector_searches (dict[str, str]): Mapping from placeholder names to search strings.
            
        Returns:
            str: SQL query with placeholders replaced by raw search strings.
        """
        final_query = sql_query
        for placeholder, search_text in vector_searches.items():
            # Format the search string for logging; you can adjust the formatting as needed.
            formatted_search_string = f"{search_text}"
            final_query = final_query.replace(f"[{placeholder}]", formatted_search_string)
        return final_query

    def execute_query(self, sql_query: str):
        """Execute SQL query."""
        return self.participant_db.run_query(sql_query)
    
    def answer_query(self, text_query: str):
        """Answer the text query by generating and executing SQL."""
        sql_output = self.write_query(text_query)
        # Inject embeddings into the SQL query by replacing the placeholders.
        final_query = self.inject_embeddings(sql_output["query"], sql_output["vector_searches"])
        # Execute the SQL query.
        sql_result = self.execute_query(final_query)
        return sql_result


if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY is None:
        raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
    
    args = parse_args()
    
    db_config = DbConfig(
        os.getenv("SUPABASE_USER"),
        os.getenv("SUPABASE_HOST"),
        os.getenv("SUPABASE_PORT"),
        os.getenv("SUPABASE_NAME"),
        os.getenv("SUPABASE_PASSWORD")
    )

    participant_db = load_participant_db(db_config, args.table_base_name)

    with open(args.prompts_config_yaml, "r") as f:
        prompt_tmpls = yaml.safe_load(f)
    text_to_sql_prompt_tmpl = prompt_tmpls[args.prompt_key]
    text_to_sql = Text2PGSQL(participant_db, text_to_sql_prompt_tmpl)
    
    # An example text query that may generate multiple vector search slots.
    text_query = "Give me the participants who are over 30 years old and have experience in data science"
    
    # Generate the SQL query with vector search placeholders and the corresponding search texts.
    sql_output = text_to_sql.write_query(text_query)
    print("Generated SQL query with placeholders:")
    print(sql_output["query"])
    print("Vector search mappings:")
    print(sql_output["vector_searches"])
    
    # Inject embeddings into the SQL query by replacing the placeholders.
    final_query = text_to_sql.inject_embeddings(sql_output["query"], sql_output["vector_searches"])
    print("Final SQL Query after embedding injection:")
    #print(final_query)
    
    # Execute the SQL query.
    sql_result = text_to_sql.execute_query(final_query)
    if sql_result["success"]:
        print("SQL Query Result:")
        pretty_print_sqlalchemy_results(sql_result['results'], max_rows=1000)
    else:
        print("SQL Query failed to execute.")
        print(sql_result["error"])
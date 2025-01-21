import os
import json
import yaml
from typing import Any, Dict, List, Union
import pandas as pd
from openai import OpenAI
import pdb

from dotenv import load_dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(current_dir, '.env'))

from participant_data import create_participant_db_df

class EntitiesExtractor:
    """
    A class for extracting structured entities from text using OpenAI models.
    Prompts are configured per-column in a YAML config file (or dictionary).
    """

    def __init__(self, config_path: str, client: OpenAI):
        """
        :param config_path: Path to a YAML or JSON config file with model & column prompts
        :param openai_api_key: Your OpenAI API key (optional). If not provided, 
                               ensure you have it set via environment variables
                               or load_dotenv, etc.
        """
        self.client = client
        self.config = self._load_config(config_path)
        self.model = self.config.get("model", "gpt-4o-mini")
        self.columns_config = self.config.get("columns", {})

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads the YAML (or JSON) config from the given path.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.endswith(".yaml") or config_path.endswith(".yml"):
                return yaml.safe_load(f)
            elif config_path.endswith(".json"):
                return json.load(f)
            else:
                raise ValueError("Unsupported config file format. Use YAML or JSON.")

    def extract_entities(
        self, 
        sp_id: Union[int, str], 
        column_name: str, 
        text: str
    ) -> List[Dict[str, Any]]:
        """
        For a single piece of text in a given column, calls OpenAI GPT to parse it 
        into structured JSON. Returns a list of parsed entities.
        """

        # If no custom prompt found for the column, fallback to a default prompt.
        column_config = self.columns_config.get(column_name)
        if column_config is None:
            raise KeyError(f"Column '{column_name}' not found in the dictionary.")
        prompt_template = column_config.get("prompt")
        user_prompt = prompt_template.format(
            sp_id=sp_id,
            column_name=column_name,
            text=text
        )

        # Prepare messages for ChatCompletion
        messages = [
            {
                "role": "system",
                "content": "You extract structured entities in JSON format, no extra text."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )
            gpt_reply = response.choices[0].message.content.strip()
            # Attempt to parse GPT's reply as JSON
            parsed_json = json.loads(gpt_reply)
            # Add full text to each extraction entry.
            [d.update({"Full Text": text}) for d in parsed_json["entities"]]
            return parsed_json.get("entities", [])
        except Exception as e:
            print(f"Error extracting entities for sp_id={sp_id}, column={column_name}: {e}")
            return []

    def extract_entities_for_df(
        self, 
        df: pd.DataFrame, 
        columns: List[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Processes each row in the given DataFrame for the specified columns,
        calls extract_entities(), and accumulates results in a dictionary 
        with one DataFrame per column.
        """
        if columns is None:
            columns = list(self.columns_config.keys())

        # Each column gets its own list of extracted entities
        results_by_column = {col: [] for col in columns}

        for _, row in df.iterrows():
            sp_id = row["SP ID"]  # adapt to your ID field name
            for col in columns:
                text_val = row.get(col, None)
                if isinstance(text_val, str) and text_val.strip() and text_val != "NA":
                    entities = self.extract_entities(sp_id, col, text_val)
                    # Append all entities for this row and column
                    results_by_column[col].extend(entities)

        # Convert each list to a DataFrame
        dfs_by_column = {}
        for col, ents in results_by_column.items():
            dfs_by_column[col] = pd.DataFrame(ents)

        return dfs_by_column

if __name__ == "__main__":

    # Load data
    target_columns = ['SP ID', 'Work Experience/Company', 'Work Experience/Designation',
       'Work Experience/Tasks', 'Work Experience/Industry',
       'Education/Qualifications', 'Education/Specialization',
       'Any Additional Skills', 'Computer Skills', 'Skills',
       'Languages', 'Gender', 'Age', 'Work Experience/From Date', 'Work Experience/To Date']
    participant_info_raw_df = pd.read_csv('data/input_participant_info_raw.csv')
    participant_db_df = create_participant_db_df(participant_info_raw_df, target_columns)

    extraction_columns = [
        "Work Experience/Designation",
        "Work Experience/Tasks",
        "Any Additional Skills",
        ]
    participant_db_df = participant_db_df[['SP ID'] + extraction_columns]
    type_list_cols = [
        "Work Experience/Designation",
        "Work Experience/Tasks",
        ]
    for col in type_list_cols:
        participant_db_df[col] = participant_db_df[col].apply(str)
    participant_db_df = participant_db_df.head(40)
    print("Participant DB DataFrame:")
    print(participant_db_df)

    client = OpenAI()
    # Instantiate our extractor with the path to our config.
    # Assuming config.yaml is in the same directory. If not, use the appropriate path.
    extractor = EntitiesExtractor(
        config_path="chatbot/preprocessing/extractor_config.yaml",
        client=client
    )

    # Let’s specify which columns to parse.
    # If you leave it None, it will parse all columns listed in config.yaml
    columns_to_parse = None
    # Extract the entities; returns a dict of DataFrames
    dfs_by_column = extractor.extract_entities_for_df(participant_db_df, columns_to_parse)

    # For each column DataFrame, do something with it
    for col_name, entities_df in dfs_by_column.items():
        entities_file_path = f"chatbot/preprocessing/data/entities_{col_name.replace(' ', '_').replace('/', '-')}.csv"
        entities_df.to_csv(entities_file_path, index=False)

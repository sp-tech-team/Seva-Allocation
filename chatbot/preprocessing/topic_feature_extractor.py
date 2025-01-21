from openai import OpenAI
import pandas as pd
import json
import os
import pdb

from participant_data import create_participant_db_df

from dotenv import load_dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(current_dir, '.env'))

client = OpenAI()

pdb.set_trace()

def extract_entities_from_text(sp_id: str, column_name: str, text: str, model="gpt-4"):
    """
    Calls the GPT model to parse 'text' into structured JSON data.
    Returns a list of parsed entities, each a Python dict.
    """

    system_msg = (
        "You are an AI that reads a piece of text describing an applicant's data "
        "and extracts the relevant entities in structured JSON."
    )

    user_prompt = f"""
    The applicant has the following data in the column {column_name}:
    \"{text}\"

    Please extract each distinct piece of information as a separate 'entity'.
    Return the result strictly in JSON format with the following structure:

    {{
      "entities": [
        {{
          "sp_id": {sp_id},
          "column": {column_name},
          "extracted_text": "<the piece of text for this entity>",
          "full_text": {text}
        }},
        ...
      ]
    }}

    Note:
    - Only return JSON, and do not include any additional text or explanation.
    """


    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    gpt_reply = response.choices[0].message.content.strip()

    try:
        parsed_json = json.loads(gpt_reply)
        return parsed_json.get("entities", [])
    except json.JSONDecodeError:
        # If parsing fails, return an empty list (or handle the error as you see fit)
        return []



def extract_all_entities_by_column(
    df: pd.DataFrame,
    text_columns=None,
    model="gpt-4o-mini"
):
    """
    Loops through each row of the DataFrame and each specified text column,
    then calls the GPT function to extract entities.
    Returns a dictionary where each key is the column name
    and each value is a list of entities extracted for that column.
    """

    if text_columns is None:
        pass

    # Initialize a dictionary to store extracted entities for each column
    all_entities_by_column = {col: [] for col in text_columns}

    for _, row in df.iterrows():
        sp_id = int(row["SP ID"])

        for col in text_columns:
            value = row[col]
            if isinstance(value, list):
                value = str(value)
            # If the cell is NaN or not a string, skip
            if not isinstance(value, str):
                continue
            if value == "NA":
                continue
            # Call GPT to parse the text
            entities = extract_entities_from_text(
                sp_id=sp_id,
                column_name=col,
                text=value,
                model=model
            )

            # Add to our dictionary for this column
            all_entities_by_column[col].extend(entities)

    return all_entities_by_column


def extract_all_entities_by_column_as_dfs(
    df: pd.DataFrame,
    text_columns=None,
    model="gpt-4o-mini"
):
    # First, extract entities by column (dict of lists)
    all_entities_by_column = extract_all_entities_by_column(df, text_columns, model)
    
    # Convert each list of entities to its own DataFrame
    dfs_by_column = {}
    for col, entities_list in all_entities_by_column.items():
        dfs_by_column[col] = pd.DataFrame(entities_list)

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
        'Work Experience/Tasks',
        "Any Additional Skills",
    ]
    participant_db_df = participant_db_df[['SP ID'] + extraction_columns]
    participant_db_df = participant_db_df.head(20)
    print("Participant DB DataFrame:")
    print(participant_db_df)
    # Extract entities for each text column
    column_entities_dfs = extract_all_entities_by_column_as_dfs(participant_db_df, text_columns=extraction_columns)
    # Save each DataFrame to a separate CSV file
    for col, entities_df in column_entities_dfs.items():
        entities_file_path = f"chatbot/preprocessing/data/entities_{col.replace(' ', '_').replace('/', '-')}.csv"
        entities_df.to_csv(entities_file_path, index=False)


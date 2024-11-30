import argparse
import os
import json
import pandas as pd

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        A namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process an input file and save the results to an output file."
    )

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Create a .env file with OpenAI and Neo4j credentials.")

    parser.add_argument("--openai-key",required=True, help="OpenAI API Key")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j URI (e.g., bolt://localhost:7687)")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j Username")
    parser.add_argument("--neo4j-password", default="your-password", help="Neo4j Password")
    parser.add_argument(
        "--vrf_data_csv",
        default='data/vrf_data.csv',
        help="Path to the vrf csv file.",
    )
    
    parser.add_argument(
        "--vrf_jobs_train_corpus_txt",
        default='data/generated_training_data/vrf_jobs_train_corpus.txt',
        help="Path to the vrf jobs training corpus.",
    )

    parser.add_argument(
        "--vrf_depts_train_corpus_txt",
        default='data/generated_training_data/vrf_depts_train_corpus.txt',
        help="Path to the vrf dept training corpus.",
    )

    parser.add_argument(
        "--prompt_config_json",
        default='configs/prompt_config.json',
        help="Path to prompt config file.",
    )

    return parser.parse_args()



def create_env_file(openai_key, neo4j_uri, neo4j_user, neo4j_password):
    # Define the content of the .env file
    env_content = \
        f"""# Environment variables
            OPENAI_API_KEY={openai_key}
            NEO4J_URI={neo4j_uri}
            NEO4J_USER={neo4j_user}
            NEO4J_PASSWORD={neo4j_password}
        """

    # Write the content to a .env file
    with open(".env", "w") as env_file:
        env_file.write(env_content)
    print(".env file created successfully!")

def create_default_index_config_file():
    # Define the data to be written to the JSON file
    data = {
        "property_graph_schema_extractor": {
            "entities": ["JOB", "SKILL"],
            "relations": ["WORKS_WITH", "RELATED_TO", "SIMILAR_TO", "USED_BY"],
            "schema": {
                "JOB": ["RELATED_TO", "SIMILAR_TO", "WORKS_WITH"],
                "SKILL": ["RELATED_TO", "USED_BY"],
            }
        }
    }
    # Define the directory and file path
    directory = "configs"
    file_path = os.path.join(directory, "index_config.json")
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    # Write the JSON data to the file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)  # Use indent=4 for pretty formatting

    print(f"JSON configuration has been written to {file_path}")

def create_training_corpus(vrf_jobs_train_corpus_file, vrf_depts_train_corpus_file, vrf_data_file):
    vrf_df = pd.read_csv(vrf_data_file)
    vrf_df = vrf_df.dropna(subset=['Job Title', 'Job Description', 'Skills/Keywords'])
    vrf_df["Job Description"] = vrf_df["Job Description"].apply(lambda x: x.replace("\n", " "))
    os.makedirs(os.path.dirname(vrf_jobs_train_corpus_file), exist_ok=True)
    os.makedirs(os.path.dirname(vrf_depts_train_corpus_file), exist_ok=True)

    vrf_df["summary"] = "The job titled: \"" + vrf_df["Job Title"] + "\" has a description: " + vrf_df["Job Description"] + ", and requires the following skills: " + vrf_df["Skills/Keywords"] + ".\n"
    vrf_corpus = "".join(vrf_df["summary"])
    with open(vrf_jobs_train_corpus_file, "w") as file:
        file.write(vrf_corpus)

    depts_df = vrf_df.groupby("Department")["Job Title"].apply(lambda x: ", ".join(x)).reset_index()
    depts_df["summary"] = "The department: \"" + depts_df["Department"] + "\" will consider the following job titles: " + depts_df["Job Title"] + ".\n"
    depts_corpus = "".join(depts_df["summary"])
    with open(vrf_depts_train_corpus_file, "w") as file:
        file.write(depts_corpus)
    print(f"Training corpus data has been written to {vrf_depts_train_corpus_file}, {vrf_jobs_train_corpus_file}")

def create_prompt_config(prompt_config_file):
    prompt_config = {
        "prompt_mode": "COMPLETE",
        "prompt_complete_file":  "configs/prompt_complete.txt"
    }
    with open(prompt_config_file, "w") as file:
        json.dump(prompt_config, file, indent=4)
    
    prompt_complete = \
"""I want you to provide job title recommendations for a set of participants based on their skills.
Each participant should get 3 job recommendations ranked by relevance to their skills
based on the job descriptions and required skills that are available in the corpus.
Please also provide 3 departments that are most relevant to the participant's skills and jobs you recommened.
Please give me the participant's number followed by the list of recommended job titles and departments in this format
"{Participant Id}/-/{Job Title rank 1},{Job Title rank 2},{Job Title rank 3}/-/{Department rank 1},{Department rank 2}, {Department rank 2}\\n".
Please do not add any other text to the response other than the id and titles.
Here is the information about the participants please do not skip a single of the 20 participants:\n"""

    with open(prompt_config["prompt_complete_file"], "w") as file:
        file.write(prompt_complete)
    print("Prompt configuration and sample have been written")

def main():
    args = parse_args()
    create_env_file(args.openai_key, args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    create_default_index_config_file()
    create_training_corpus(args.vrf_jobs_train_corpus_txt, args.vrf_depts_train_corpus_txt, args.vrf_data_csv)
    create_prompt_config(args.prompt_config_json)

if __name__ == "__main__":
    main()
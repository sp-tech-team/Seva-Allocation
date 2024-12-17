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

    return parser.parse_args()



def create_env_file(openai_key):
    """
    Create a .env file with the OpenAI and Neo4j credentials.
    
    Args:
    openai_key (str): The OpenAI API key.
    """
    # Define the content of the .env file
    env_content = \
        f"""# Environment variables
            OPENAI_API_KEY={openai_key}
        """

    # Write the content to a .env file
    with open(".env", "w") as env_file:
        env_file.write(env_content)
    print(".env file created successfully!")

def main():
    args = parse_args()
    create_env_file(args.openai_key, args.neo4j_uri, args.neo4j_user, args.neo4j_password)

if __name__ == "__main__":
    main()
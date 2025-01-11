def create_env_file(openai_key):
    # Define the content of the .env file
    env_content = \
        f"""# Environment variables
            OPENAI_API_KEY={openai_key}
        """

    # Write the content to a .env file
    with open(".env", "w") as env_file:
        env_file.write(env_content)
    print(".env file created successfully!")

openai_key = 'put_your_openai_key_here'
create_env_file(openai_key)
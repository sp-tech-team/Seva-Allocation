import gradio as gr
import argparse
import os
import yaml
from database.participant_pg_database import DbConfig, load_participant_db
from chatbot.pg_text_to_sql import Text2PGSQL
from chatbot.pg_chatbot import ChatbotPipeline

# May need to remove this for huggingface spaces
from dotenv import load_dotenv
load_dotenv(".env")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        A namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--table_base_name',
        type=str,
        help='Name of the table to query',
        default='participants_mock2'
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
    parser.add_argument(
        '--make_public',
        action='store_true',
        help='Whether to make the Gradio app public')

    return parser.parse_args()

args = parse_args()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
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
pipeline = ChatbotPipeline(text_to_sql)

# === Define Chatbot Logic ===
def chat_with_bot(user_input, chat_history):
    """
    Handles user input by fetching relevant context using the retriever
    and generating a response using the LLM.
    """
    try:
        pretty_sql_results = pipeline.chatbot(user_input)
        chat_history.append(("User", user_input))
        chat_history.append(("Bot", pretty_sql_results))

        return "", chat_history  # Return updated chat history and clear input box
    except Exception as e:
        return "", chat_history + [("Bot", f"Error: {str(e)}")]

# === Gradio Blocks Interface ===
def create_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("# Seva Text 2 SQL Chatbot")

        # Login UI Components
        with gr.Row():
            password_input = gr.Textbox(placeholder="Enter password...", type="password", label="Password")
            login_btn = gr.Button("Login")

        # Chatbot UI Components
        with gr.Row(visible=False) as chat_row:
            chatbot = gr.Chatbot()
        
        with gr.Row(visible=False) as input_row:
            user_input = gr.Textbox(placeholder="Ask me anything...", label="Your Query")
            submit_btn = gr.Button("Send")

        # Clear chat button
        with gr.Row(visible=False) as clear_row:
            clear_btn = gr.Button("Clear Chat")

        # Define callbacks
        def login(password):
            if password == os.getenv("GRADIO_APP_PASSWORD"):
                chat_row.visible = True
                input_row.visible = True
                clear_row.visible = True
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        login_btn.click(
            login, 
            inputs=[password_input], 
            outputs=[password_input, chat_row, input_row, clear_row]
        )

        submit_btn.click(
            chat_with_bot, 
            inputs=[user_input, chatbot], 
            outputs=[user_input, chatbot]
        )
        clear_btn.click(
            lambda: None, inputs=None, outputs=chatbot
        )

    return demo

# === Launch the App ===
if __name__ == "__main__":
    gradio_app = create_interface()
    if args.make_public:
        gradio_app.launch(share=True)
    else:
        gradio_app.launch()
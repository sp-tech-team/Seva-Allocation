import gradio as gr
from chatbot import initialize_pipeline, db_selector_query, prompt_tmpl
import argparse
import os


def inference_parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        A namespace with parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process an input file and save the results to an output file."
    )

    parser.add_argument(
        "--num_retrievals",
        default=10,
        type=int,
        help="Number of retrievals from db.",
    )

    parser.add_argument(
        "-participant_pinecone_index_name",
        default='participant-test-local',
        help="Path to vector store dbs.",
    )

    parser.add_argument(
        "-vrf_pinecone_index_name",
        default='vrf-test-local',
        help="Path to vector store dbs.",
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        dest='verbose',  # Default False
        help="Enable verbose logging. (default is disabled)"
    )

    return parser.parse_args()

args = inference_parse_args()
retrievers, llm = initialize_pipeline(args.participant_pinecone_index_name, args.vrf_pinecone_index_name, args.num_retrievals)

# === Define Chatbot Logic ===
def chat_with_bot(user_input, chat_history):
    """
    Handles user input by fetching relevant context using the retriever
    and generating a response using the LLM.
    """
    try:
        db_selector_response = llm.complete(db_selector_query + user_input)
        if db_selector_response.text not in retrievers:
            raise ValueError(f"Invalid database selector response: {db_selector_response.text}")
        nodes = retrievers[db_selector_response.text].retrieve(user_input)
        context_str = "\n".join(node.get_content() for node in nodes)
        query = prompt_tmpl.format(context_str=context_str, query_str=user_input)
        response = llm.complete(query)

        # Update chat history
        bot_response = response.text
        chat_history.append(("User", user_input))
        chat_history.append(("Bot", bot_response))

        return "", chat_history  # Return updated chat history and clear input box
    except Exception as e:
        return "", chat_history + [("Bot", f"Error: {str(e)}")]

# === Gradio Blocks Interface ===
def create_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("# Chatbot with LlamaIndex RAG Pipeline (Optimized)")

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
    gradio_app.launch()

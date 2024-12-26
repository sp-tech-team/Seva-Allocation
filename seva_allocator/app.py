import gradio as gr
from chatbot import initialize_pipeline
import argparse


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
        default='participant-test',
        help="Path to vector store dbs.",
    )

    parser.add_argument(
        "-vrf_pinecone_index_name",
        default='vrf-test',
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
        # Retrieve relevant nodes
        retrieved_nodes = retrievers["PARTICIPANT_DB"].retrieve(user_input)

        # Concatenate the node content to form the context
        context = "\n\n".join(node.get_text() for node in retrieved_nodes)

        # Prepare the prompt for the LLM
        prompt = f"Context:\n{context}\n\nQuestion: {user_input}\n\nAnswer:"

        # Generate a response using the LLM
        response = llm.complete(prompt)

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

        # Chatbot UI Components
        with gr.Row():
            chatbot = gr.Chatbot()
        
        with gr.Row():
            user_input = gr.Textbox(placeholder="Ask me anything...", label="Your Query")
            submit_btn = gr.Button("Send")

        # Clear chat button
        with gr.Row():
            clear_btn = gr.Button("Clear Chat")

        # Define callbacks
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

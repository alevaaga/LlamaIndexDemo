from collections import defaultdict

import gradio as gr
from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.base import DEFAULT_OPENAI_MODEL

import crayon
from crayon.finance_chatbot import create_chat_engine as create_chat_engine_full
from crayon.finance_chatbot_hello_world import create_chat_engine as create_chat_engine_hello_world
from crayon.finance_chatbot_simple import create_chat_engine as create_chat_engine_simple

#
# PyCharm debugger issues:
# As a workaround I disabled this settings -> Help | Find Action | Registry | python.debug.asyncio.repl
#

backends = {
    "hello_world": create_chat_engine_hello_world,
    "simple": create_chat_engine_simple,
    "full": create_chat_engine_full
}

selected_backend = "full"
chat_engines = defaultdict(lambda: backends[selected_backend]())


def get_chat_engine() -> BaseChatEngine:
    global selected_backend
    chat_engine = chat_engines[selected_backend]
    return chat_engine


def select_backend(backend: str):
    global selected_backend
    selected_backend = backend


def main():
    def predict(message, history):
        global selected_backend
        nonlocal citations
        chat_history = []
        for msg, reply in history:
            if msg.strip() != "":
                chat_history.append(ChatMessage(content=msg, role=MessageRole.USER))
            if reply.strip() != "":
                chat_history.append(ChatMessage(content=reply, role=MessageRole.ASSISTANT))

        chat_engine = get_chat_engine()
        reply = chat_engine.chat(message, chat_history=chat_history)

        citation_str = "<h1>Citations</h1></br>"
        for src in reply.source_nodes:
            if not src.score:
                continue
            citation_str += f"<H2>Score: {src.score:.2f}</H2>"
            citation_str += f"<H3>Filename: {src.metadata['filename']}</H3>"
            citation_str += f"<p>{src.text}</p>"

        citations = citation_str
        return str(reply)

    citations = "<none>"

    def get_citation(*args):
        return citations

    chatbot = gr.Chatbot(height=500)

    chat_interface = gr.ChatInterface(
        predict,
        chatbot=chatbot,
        textbox=gr.Textbox(placeholder="Ask me something", container=False, scale=7, lines=4),
        theme="soft",
        examples=[
            ["Compare the profitability of Crayon and SoftwareOne, year of year, from 2019 through 2023"],
            ["How much did Crayon's gross profit increase from 2018 to 2023?"],
            ["What was Crayon's total revenue in 2023?"],
            ["How much did it grow from 2022?"],
            ["In what year did Crayon have it's highest earning pr. share?"],
            ["In what year did Crayon have it's lowest earning pr. share?"],
            ["What was Crayon's total revenue in 2017 and 2023?"],
            ["What was Uber's highest cost in 2020?"],
            ["Did it go up or down in 2021?"],
            ["In what year did SoftwareOne have it's highest earning pr. share?"],
            ["Compare the profitability of Crayon and SoftwareOne, year of year, from 2019 through 2023"],
        ],
        cache_examples=False,
        retry_btn="Regenerate",
        undo_btn="Delete Previous",
        clear_btn="Clear",
    )

    with gr.Blocks(fill_height=True) as demo:
        with gr.Row():
            with gr.Column():
                chat_interface.render()
                backend_radio = gr.Radio(choices=[
                    ("Hello, World", "hello_world"),
                    ("Index pr. Company", "simple"),
                    ("Index pr. Document", "full")
                ], label="Backend", value="full")
                backend_radio.select(fn=select_backend, inputs=backend_radio)
            with gr.Column():
                # output = gr.TextArea(label="Citations")
                output = gr.HTML()
                chat_interface.chatbot.change(fn=get_citation, outputs=output)

    demo.launch()


if __name__ == '__main__':
    crayon.STORAGE_ROOT = "storage/Finance"
    crayon.CACHE_ROOT = "storage/cache"

    llm = OpenAI(model=DEFAULT_OPENAI_MODEL)

    # llm = OpenAI(
    #     api_base="http://wizzo.akhbar.home:5000/v1",
    #     api_key="sk-ollama",
    #     model="gpt-4-turbo",
    #     temperature=1.0,
    #     timeout=180,
    #     verbose=True
    # )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager
    main()

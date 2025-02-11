from pathlib import Path
from typing import Any, Dict, List

import gradio as gr
from httpx import _config
from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

import crayon
from crayon.chatbots.finance_chatbot import create_chat_engine

#
# PyCharm debugger issues:
# As a workaround I disabled this settings -> Help | Find Action | Registry | python.debug.asyncio.repl
#
_config.DEFAULT_TIMEOUT_CONFIG = _config.Timeout(timeout=500)

def get_chat_engine() -> BaseChatEngine:
    chat_engine = create_chat_engine()
    return chat_engine


def get_llm(temperature: float) -> OpenAI:
    llm = OpenAI(
        api_base="http://wizzo.akhbar.home:5000/v1",
        api_key="sk-ollama",
        model="gpt-4-turbo",
        temperature=temperature,
        timeout=180,
        verbose=True
    )
    return llm


def main():
    def predict(message: Dict[str, Any], history: List, system_prompt: str, temperature: float):
        files = []
        for msg in history:
            if msg['role'] == "user" and isinstance(msg['content'], tuple):
                files.append(msg['content'][0])

        if isinstance(message, dict):
            for file in message["files"]:
                files.append(file)
            query = message["text"]
        else:
            query = message

        llm = get_llm(temperature)
        chat_engine = get_chat_engine()

        system_prompt_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
        chat_history = [system_prompt_msg] + [ChatMessage(role=MessageRole.USER if m["role"] == "user" else MessageRole.ASSISTANT, content=m["content"]) for m in history]
        chat_engine.llm = llm
        reply = chat_engine.chat(query, chat_history=chat_history)

        citation_str = "<h1>Citations</h1></br>"
        for src in reply.source_nodes:
            if not src.score:
                continue
            citation_str += f"<H2>Score: {src.score:.2f}</H2>"
            citation_str += f"<H3>Filename: {src.metadata['filename']}</H3>"
            citation_str += f"</br></br>"
            citation_str += f"<p>{src.text}</p>"
            citation_str += f"</br></br>"

        return str(reply), gr.Markdown(citation_str)

    with gr.Blocks(fill_height=True) as demo:
        citations_component = gr.Markdown(render=False)
        system_prompt = gr.Textbox(
            value="You are a financial analyst at a large investment bank. Use the provided tools and context to answer the users query.",
            label="System Prompt",
            lines=10,
            render=False,
        )

        with gr.Tab("Chatbot"):
            with gr.Row():
                with gr.Column():
                    chat_interface = gr.ChatInterface(
                        predict,
                        type="messages",
                        chatbot=gr.Chatbot(height=500, type="messages"),
                        textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt"]),
                        additional_inputs=[
                            system_prompt,
                            gr.Slider(minimum=0, maximum=1, step=0.01, label="Temperature", value=0.5, render=False),
                        ],
                        additional_outputs=[citations_component],
                        editable=True,
                        theme="soft",
                        cache_examples=False,
                        run_examples_on_click=False,
                        save_history=True,
                    )
                    ex = gr.Examples([
                        ["Compare the profitability of Crayon and SoftwareOne, year of year, from 2019 through 2023. Explain your reasoning."],
                        ["How much did Crayon's gross profit increase from 2018 to 2023?"],
                        ["What was Crayon's total revenue in 2023?"],
                        ["How much did it grow from 2022?"],
                        ["In what year did Crayon have it's highest earning pr. share?"],
                        ["In what year did Crayon have it's lowest earning pr. share?"],
                        ["What was Crayon's total revenue in 2017 and 2023?"],
                        ["What was Uber's highest cost in 2020?"],
                        ["Did it go up or down in 2021?"],
                        ["In what year did SoftwareOne have it's highest earning pr. share?"],
                    ], inputs=chat_interface.textbox)
                with gr.Column():
                    citations_component.render()
        with gr.Tab("Evaluation"):
            gr.Text("Evaluation")

    demo.launch(server_name="0.0.0.0", pwa=True)


if __name__ == '__main__':
    print(f"Current directory: {Path('.').resolve()}")
    print("List of files:")
    print(list(Path('.').glob("*")))
    crayon.STORAGE_ROOT = "storage/Finance"
    crayon.CACHE_ROOT = "storage/cache"

    # llm = OpenAI(model=DEFAULT_OPENAI_MODEL)

    llm = OpenAI(
        api_base="http://wizzo.akhbar.home:5000/v1",
        api_key="sk-ollama",
        model="gpt-4-turbo",
        temperature=0.5,
        timeout=180,
        verbose=True
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    Settings.callback_manager = callback_manager
    main()

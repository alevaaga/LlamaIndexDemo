from collections import defaultdict
from pathlib import Path

from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import gradio as gr

def answer(message, history):
    files = []
    for msg in history:
        if msg['role'] == "user" and isinstance(msg['content'], tuple):
            files.append(msg['content'][0])
    for file in message["files"]:
        files.append(file)

    documents = SimpleDirectoryReader(input_files=files).load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    return str(query_engine.query(message["text"]))

demo = gr.ChatInterface(
    answer,
    type="messages",
    title="Llama Index RAG Chatbot",
    description="Upload any text or pdf files and ask questions about them!",
    textbox=gr.MultimodalTextbox(file_types=[".pdf", ".txt"]),
    multimodal=True
)

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

demo.launch()

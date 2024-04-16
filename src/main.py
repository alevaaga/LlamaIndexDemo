import logging
import sys

import gradio as gr
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings
from llama_index.core.base.llms.types import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.base import DEFAULT_OPENAI_MODEL

import crayon
from crayon.finance_chatbot import build_tools, load_indices

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

crayon.STORAGE_ROOT = "storage/Finance"
crayon.CACHE_ROOT = "storage/cache"

llm = OpenAI(model=DEFAULT_OPENAI_MODEL)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model


def predict(message, history):
    chat_history = []
    for msg, reply in history:
        chat_history.append(ChatMessage(content=msg, role="user"))
        chat_history.append(ChatMessage(content=reply, role="assistant"))
    reply = agent.chat(message, chat_history=chat_history)
    return str(reply)


companies_index_set = load_indices()
all_tools = build_tools(companies_index_set)

agent = OpenAIAgent.from_tools(tools=all_tools, llm=Settings.llm, verbose=True)

gr.ChatInterface(
    predict,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me a yes something", container=False, scale=7, lines=4),
    title="Finance Chatbot",
    description="Ask me something about Crayon, Software One or Uber's financials!",
    theme="soft",
    examples=["What was Crayon's total revenue in 2023?", "How much did it grow from 2022?", "What was Uber's highest cost in 2020?", "Did it go up or down in 2021?"],
    cache_examples=False,
    retry_btn="Regenerate",
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()

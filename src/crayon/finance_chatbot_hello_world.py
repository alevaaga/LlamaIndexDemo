import logging
import sys
from typing import Tuple

import llama_index
from llama_index.core import Settings, load_indices_from_storage, ServiceContext
from llama_index.core.callbacks import LlamaDebugHandler
from llama_index.core.chat_engine.types import ChatMode, BaseChatEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.base import DEFAULT_OPENAI_MODEL

import crayon

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def load_indices(db_name: str) -> BaseIndex:
    print("Initializing context.....")
    storage_context = crayon.get_storage_context_filesystem(db_name=db_name)
    service_context = ServiceContext.from_defaults(llm=Settings.llm, embed_model=Settings.embed_model)

    print("Loading indices.....")
    all_indices = load_indices_from_storage(storage_context=storage_context, service_context=service_context, index_ids=None)
    assert len(all_indices) == 1, f"The number of indices should be 1 but got {len(all_indices)}!"

    print("Indices loaded!")
    return all_indices[0]


def repl_query_engine():
    index = load_indices(db_name="HelloWorldIndex")
    query_engine = index.as_query_engine()

    while True:
        text_input = input("User: ")
        if text_input == "exit":
            break
        response = query_engine.query(text_input)
        print(f"Agent: {response}")


def dump_memory(memory: ChatMemoryBuffer):
    messages = memory.get_all()
    for m in messages:
        print(f"{m.role.upper()}: {m.content}")


def create_chat_engine()-> BaseChatEngine:
    index = load_indices(db_name="HelloWorldIndex")

    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    chat_engine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        memory=memory,
        system_prompt=(
            "You are a chatbot, able to have normal interactions, as well as "
            "answer questions about Crayon, SoftwareOne and Uber."
        ),
    )
    return chat_engine


def repl_chat_engine():
    chat_engine, memory = create_chat_engine()
    while True:
        text_input = input("User: ")
        if text_input == "exit":
            dump_memory(memory)
            break
        response = chat_engine.chat(text_input)
        print(f"Agent: {response}")


if __name__ == '__main__':
    print("Staring Chatbot Hello, World")

    crayon.STORAGE_ROOT = "storage/Finance"
    crayon.CACHE_ROOT = "storage/cache"

    llama_index.core.global_handler = LlamaDebugHandler(print_trace_on_end=True)
    # llm = FunctionalOpenAILike(
    #     # model="dolphin-mixtral:latest",
    #     model="nous-hermes2-mixtral:large-ctx",
    #     api_base="http://ws239.akhbar.home:5000/v1",
    #     api_key="sk-ollama",
    #     temperature=0.1,  # Default 0.7
    #     top_p=0.9,  # Default 0.9
    #     timeout=120,
    #     # max_tokens=4096,
    #     # context_window=16384,
    #     is_function_calling_model=True,
    #     is_chat_model=True,
    # )
    llm = OpenAI(model=DEFAULT_OPENAI_MODEL)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    repl_query_engine()
    repl_chat_engine()

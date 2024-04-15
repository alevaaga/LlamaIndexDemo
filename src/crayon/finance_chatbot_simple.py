import logging
import re
import sys
from typing import List, Dict, Any

import llama_index
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings, load_indices_from_storage, ServiceContext
from llama_index.core.callbacks import LlamaDebugHandler
from llama_index.core.indices.base import BaseIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.base import DEFAULT_OPENAI_MODEL

import crayon

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def build_tools(companies_index_set: Dict[str, Dict[Any, BaseIndex]]) -> List[QueryEngineTool]:
    print("Building tools.....")
    all_tools = []
    for company in companies_index_set.keys():
        index_set = companies_index_set[company]

        individual_query_engine_tools = [
            QueryEngineTool(
                query_engine=index_set[index_name].as_query_engine(),
                metadata=ToolMetadata(
                    name=f"vector_index_{company}_{company}",
                    description=f"useful for when you want to answer queries about the {index_name} SEC 10-K for {company}",
                ),
            )
            for index_name in index_set.keys()
        ]
        all_tools.extend(individual_query_engine_tools)

    print("Tools built!")
    return all_tools


def load_indices(db_name: str) -> Dict[str, Dict[Any, BaseIndex]]:
    print("Initializing context.....")
    storage_context = crayon.get_storage_context_filesystem(db_name=db_name)
    service_context = ServiceContext.from_defaults(llm=Settings.llm, embed_model=Settings.embed_model)

    print("Loading indices.....")
    all_indices = load_indices_from_storage(storage_context=storage_context, service_context=service_context, index_ids=None)

    companies_index_set = {}
    for index in all_indices:
        index_id = index.index_id
        matches = re.match(r"(\w+)_\w*_VectorIndex", index_id)
        company_name = matches.group(1)
        company_index_set = companies_index_set.get(company_name, {})
        company_index_set[company_name] = index
        companies_index_set[company_name] = company_index_set

    print("Indices loaded!")
    return companies_index_set


def main():
    companies_index_set = load_indices(db_name="SimpleIndex")
    all_tools = build_tools(companies_index_set)

    agent = OpenAIAgent.from_tools(tools=all_tools, llm=Settings.llm, verbose=True)

    while True:
        text_input = input("User: ")
        if text_input == "exit":
            break
        response = agent.chat(text_input)
        print(f"Agent: {response}")


if __name__ == '__main__':
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

    main()

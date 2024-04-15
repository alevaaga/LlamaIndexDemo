import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import llama_index
from llama_index.core import Settings
from llama_index.core.callbacks import LlamaDebugHandler
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.base import DEFAULT_OPENAI_MODEL

import crayon
from crayon.ingestion.finance.ingestion import load_documents
from crayon.ingestion.indexing.strategies import IndexStrategies

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def get_transformations():
    transformations = [
        SentenceSplitter.from_defaults(chunk_size=1024, chunk_overlap=20, callback_manager=Settings.callback_manager),
        Settings.embed_model,
    ]

    return transformations


def build_indices(base_dir: str, db_name: str) -> Dict[str, Dict[Any, BaseIndex]]:
    print(f"Loading files and creating index.....")
    company_paths = filter(lambda p: p.is_dir(), Path(base_dir).glob("*"))
    index_set = {}
    for company_path in company_paths:
        company_name = company_path.name
        print(f"Loading documents for {company_path}.....")
        company_docs = load_documents(base_dir=company_path)

        print(f"Creating Index for {company_path}.....")
        index_strategy = IndexStrategies["simple"].create(db_name=db_name)
        company_index_set = index_strategy.build_index(
            namespace=company_name,
            nodes=company_docs,
            transforms=get_transformations()
        )
        index_set[company_name] = company_index_set
    return index_set


def build_tools(companies_index_set: Dict[str, Dict[Any, BaseIndex]]) -> List[QueryEngineTool]:
    all_tools = []
    for company in companies_index_set:
        index_set = companies_index_set[company]

        individual_query_engine_tools = [
            QueryEngineTool(
                query_engine=index_set[index_name].as_query_engine(),
                metadata=ToolMetadata(
                    name=f"vector_index_{company}_{company}",
                    description=f"useful for when you want to answer queries about SEC 10-K for {company}",
                ),
            )
            for index_name in index_set.keys()
        ]
        company_tools = individual_query_engine_tools
        all_tools.extend(company_tools)
    return all_tools


def main(base_dir: str):
    companies_index_set = build_indices(base_dir=base_dir, db_name="SimpleIndex")
    print("Indices built!")


if __name__ == '__main__':
    crayon.STORAGE_ROOT = "storage/Finance"
    crayon.CACHE_ROOT = "storage/cache"

    input_dir = "data/Knowledgebase/Finance"

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

    main(base_dir=input_dir)

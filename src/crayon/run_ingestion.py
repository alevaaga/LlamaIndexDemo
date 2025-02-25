import logging
import sys
from pathlib import Path
from typing import Dict, Any

import llama_index
from llama_index.core import Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.callbacks import LlamaDebugHandler
from llama_index.core.indices.base import BaseIndex
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
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


def build_indices(base_dir: str, db_name: str, llm: BaseLLM, embed_model: BaseEmbedding) -> Dict[str, Dict[Any, BaseIndex]]:
    print(f"Loading files and creating index.....")
    company_paths = filter(lambda p: p.is_dir(), Path(base_dir).glob("*"))
    index_set = {}
    for company_path in company_paths:
        company_name = company_path.name
        print(f"Loading documents for {company_name}.....")
        company_docs = load_documents(base_dir=company_path)
        storage_context = crayon.get_storage_context_filesystem(db_name=db_name)

        print(f"Creating Index for {company_path}.....")
        index_strategy = IndexStrategies["by_year"].create(db_name=db_name)
        company_index_set = index_strategy.build_index(
            llm=llm,
            embed_model=embed_model,
            namespace=company_name,
            storage_context=storage_context,
            nodes=company_docs,
            transforms=get_transformations()
        )
        index_set[company_name] = company_index_set
    return index_set


def main(base_dir: str, llm: BaseLLM, embed_model: BaseEmbedding):
    companies_index_set = build_indices(base_dir=base_dir, db_name="MultiIndex", llm=llm, embed_model=embed_model)
    print("Indices built!")


if __name__ == '__main__':
    crayon.STORAGE_ROOT = "storage/Finance"
    crayon.CACHE_ROOT = "storage/cache"

    input_dir = "data/Knowledgebase/Finance"

    llama_index.core.global_handler = LlamaDebugHandler(print_trace_on_end=True)
    llm = OpenAI(model=DEFAULT_OPENAI_MODEL)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    main(base_dir=input_dir, llm=llm, embed_model=embed_model)

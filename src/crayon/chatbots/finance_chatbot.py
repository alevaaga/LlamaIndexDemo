import re
from typing import List, Dict, Any, Optional

import llama_index
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings, load_indices_from_storage
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.callbacks import LlamaDebugHandler
from llama_index.core.chat_engine.types import BaseChatEngine
from llama_index.core.indices.base import BaseIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.question_gen import LLMQuestionGenerator
from llama_index.core.tools import QueryEngineTool
from llama_index.core.tools import ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.base import DEFAULT_OPENAI_MODEL

import crayon


def build_tools(companies_index_set: Dict[str, Dict[Any, BaseIndex]]) -> List[QueryEngineTool]:
    print("Building tools.....")
    all_tools = []
    for company in companies_index_set.keys():
        index_set = companies_index_set[company]

        individual_query_engine_tools = [
            QueryEngineTool(
                query_engine=index_set[year].as_query_engine(similarity_top_k=5),
                metadata=ToolMetadata(
                    name=f"vector_index_{company}_{year}",
                    description=f"useful for when you want to answer queries about the {year} SEC 10-K for {company}",
                ),
            )
            for year in index_set.keys()
        ]

        sub_query_engine_tool = QueryEngineTool(
            query_engine=SubQuestionQueryEngine.from_defaults(
                query_engine_tools=individual_query_engine_tools,
                question_gen=LLMQuestionGenerator.from_defaults(llm=Settings.llm),
                llm=Settings.llm
            ),
            metadata=ToolMetadata(
                name=f"{company}_sub_question_query_engine",
                description=f"useful for when you want to answer queries that require analyzing multiple SEC 10-K documents for {company}",
            ),
        )
        company_tools = individual_query_engine_tools + [sub_query_engine_tool]
        all_tools.extend(company_tools)

    print("Tools built!")
    return all_tools


def load_indices() -> Dict[str, Dict[Any, Any]]:
    print("Initializing context.....")
    storage_context = crayon.get_storage_context_filesystem(db_name="MultiIndex")

    print("Loading indices.....")
    all_indices = load_indices_from_storage(storage_context=storage_context, index_ids=None)

    companies_index_set = {}
    for index in all_indices:
        index_id = index.index_id
        matches = re.match(r"(\w+)_(\d*)_VectorIndex", index_id)
        company_name = matches.group(1)
        year = int(matches.group(2))
        company_index_set = companies_index_set.get(company_name, {})
        company_index_set[year] = index
        companies_index_set[company_name] = company_index_set

    print("Indices loaded!")
    return companies_index_set


def create_chat_engine(llm: Optional[BaseLLM] = None) -> BaseChatEngine:
    llm = llm or Settings.llm

    companies_index_set = load_indices()
    all_tools = build_tools(companies_index_set)

    memory = ChatMemoryBuffer.from_defaults(token_limit=10000)
    agent = OpenAIAgent.from_tools(tools=all_tools, llm=llm, memory=memory, verbose=True)
    return agent


def main():
    agent, memory = create_chat_engine()
    while True:
        text_input = input("User: ")
        if text_input == "exit":
            break
        response = agent.chat(text_input)
        print(f"Agent: {response}")


if __name__ == '__main__':
    print("Staring Chatbot Full")
    crayon.STORAGE_ROOT = "storage/Finance"
    crayon.CACHE_ROOT = "storage/cache"

    llama_index.core.global_handler = LlamaDebugHandler(print_trace_on_end=True)
    llm = OpenAI(model=DEFAULT_OPENAI_MODEL)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    main()

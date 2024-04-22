import logging
import sys
import uuid
from pathlib import Path
from typing import Dict

import requests
from llama_index.core import SimpleDirectoryReader, Document, Settings, VectorStoreIndex
from llama_index.core import SummaryIndex
from llama_index.core.callbacks import LlamaDebugHandler, CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from functional_openai_like.openai_like import FunctionalOpenAILike
from retrieval import get_storage_context_filesystem

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


wiki_titles = ["Michael Jordan", "Elon Musk", "Richard Branson", "Rihanna"]
wiki_metadatas = {
    "Michael Jordan": {
        "category": "Sports",
        "country": "United States",
    },
    "Elon Musk": {
        "category": "Business",
        "country": "United States",
    },
    "Richard Branson": {
        "category": "Business",
        "country": "UK",
    },
    "Rihanna": {
        "category": "Music",
        "country": "Barbados",
    },
}


def download_data() -> Dict[str, Document]:
    data_path = Path("data/wikipedia")

    for title in wiki_titles:
        output_filename = data_path / f"{title}.txt"
        if not output_filename.exists():
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "titles": title,
                    "prop": "extracts",
                    # 'exintro': True,
                    "explaintext": True,
                },
            ).json()
            page = next(iter(response["query"]["pages"].values()))
            wiki_text = page["extract"]

            if not data_path.exists():
                Path.mkdir(data_path)

            with open(output_filename, "w") as fp:
                fp.write(wiki_text)

    # Load all wiki documents
    docs_dict = {}
    for wiki_title in wiki_titles:
        doc = SimpleDirectoryReader(
            input_files=[f"{data_path}/{wiki_title}.txt"]
        ).load_data()[0]

        doc.metadata.update(wiki_metadatas[wiki_title])
        docs_dict[wiki_title] = doc
    return docs_dict


def build_retrievers(docs_dict: Dict[str, Document]):
    storage_context = get_storage_context_filesystem(db_name="WikipediaParent")

    sub_chunk_sizes = [128, 256]
    base_splitter = SentenceSplitter(chunk_size=1024)

    all_nodes = []
    for i, wiki_title in enumerate(wiki_titles):
        wiki_doc = docs_dict[wiki_title]
        wiki_doc.id_ = f"WIKI_DOC_{i}"
        base_nodes = base_splitter.get_nodes_from_documents([wiki_doc])
        for bn in base_nodes:
            bn.metadata["chunk_size"] = 1024

        for base_node in base_nodes:
            for chunk_size in sub_chunk_sizes:
                n = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=20)
                sub_nodes = n.get_nodes_from_documents([base_node])
                for sn in sub_nodes:
                    sn.metadata["chunk_size"] = chunk_size

                sub_inodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                all_nodes.extend(sub_inodes)

            # also add original node to node
            parent_index_node = IndexNode.from_text_node(base_node, base_node.node_id)
            all_nodes.append(parent_index_node)

    vector_index = VectorStoreIndex(nodes=all_nodes, storage_context=storage_context)
    # all_nodes_dict = {n.node_id: n for n in all_nodes} # This is so UGLY!!!

    recursive = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_index.as_retriever()},
        node_dict=storage_context.docstore.docs,
        verbose=True,
    )

    return recursive


def repl():
    docs = download_data()
    recursive_retriever = build_retrievers(docs)
    query_engine = RetrieverQueryEngine.from_args(recursive_retriever)

    while True:
        text_input = input("Query: ")
        if text_input == "exit":
            break
        response = query_engine.query(text_input)
        print(f"Result: {response}")



if __name__ == '__main__':
    # llama_index.core.global_handler = LlamaDebugHandler(print_trace_on_end=True)
    llm = FunctionalOpenAILike(
        # model="dolphin-mixtral:latest",
        model="nous-hermes2-mixtral:large-ctx",
        api_base="http://ws239.akhbar.home:5000/v1",
        api_key="sk-ollama",
        temperature=0.1,  # Default 0.7
        top_p=0.9,  # Default 0.9
        timeout=120,
        # max_tokens=4096,
        # context_window=16384,
        is_function_calling_model=True,
        is_chat_model=True,
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = llm
    Settings.embed_model = embed_model

    repl()


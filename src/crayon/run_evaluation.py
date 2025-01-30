import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict

import nest_asyncio
from llama_index.core import Document
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.evaluation import BatchEvalRunner
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
)
from llama_index.core.evaluation import DatasetGenerator, QueryResponseDataset
from llama_index.core.evaluation.eval_utils import (
    get_responses,
    get_results_df,
)
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.llms import LLM
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.base import DEFAULT_OPENAI_MODEL
from llama_index.llms.openai_like import OpenAILike

from crayon import get_storage_context_filesystem
from crayon.ingestion.chunking.strategies import ChunkingStrategies
from crayon.ingestion.finance import get_wikipedia_finance

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

eval_dataset_filename = "storage/finance/evaluation/evaluation_results.json"
eval_rag_dataset_filename = "storage/finance/evaluation/evaluation_rag_dataset.json"


def generate_eval_dataset(llm, nodes):
    root_nodes = ChunkingStrategies["simple"](chunk_size=1024, chunk_overlap=0, debug=True).chunk(nodes)

    # NOTE: run this if the dataset isn't already saved
    # Note: we only generate from the first 20 nodes, since the rest are references
    eval_llm = llm

    if not Path(eval_dataset_filename).exists():
        dataset_generator = DatasetGenerator(
            root_nodes,
            llm=eval_llm,
            show_progress=True,
            num_questions_per_chunk=3,
        )

        print("Generating evaluation dataset...")
        eval_dataset = asyncio.run(dataset_generator.agenerate_dataset_from_nodes(num=60))
        Path("storage/finance/evaluation").mkdir(parents=True, exist_ok=True)
        eval_dataset.save_json(eval_dataset_filename)
        print("Evaluation dataset saved!")

    if not Path(eval_rag_dataset_filename).exists():
        print("Generating RAG dataset...")
        rag_dataset_generator = RagDatasetGenerator.from_documents(documents=root_nodes, num_questions_per_chunk=3)
        rag_dataset = asyncio.run(rag_dataset_generator.agenerate_questions_from_nodes())
        rag_dataset.save_json(eval_rag_dataset_filename)
        print("RAG dataset saved!")


def get_retrievers(docs: List[Document]) -> Dict[str, BaseRetriever]:
    chunks_1024 = ChunkingStrategies["simple"](chunk_size=1024, chunk_overlap=0, debug=True).chunk(docs)
    chunks_512 = ChunkingStrategies["simple"](chunk_size=512, chunk_overlap=0, debug=True).chunk(docs)
    chunks_256 = ChunkingStrategies["simple"](chunk_size=256, chunk_overlap=0, debug=True).chunk(docs)
    chunks_128 = ChunkingStrategies["simple"](chunk_size=128, chunk_overlap=0, debug=True).chunk(docs)

    chunks_parent = ChunkingStrategies["parent"](base_size=1024, sub_chunk_sizes=[512, 256, 128], chunk_overlap=20, debug=True).chunk(docs)
    chunks_hierarchical = HierarchicalNodeParser.from_defaults().get_nodes_from_documents(docs)

    storage_context = get_storage_context_filesystem(db_name="Evaluation")

    index_1024 = VectorStoreIndex(nodes=chunks_1024, storage_context=storage_context)
    index_512 = VectorStoreIndex(nodes=chunks_512, storage_context=storage_context)
    index_256 = VectorStoreIndex(nodes=chunks_256, storage_context=storage_context)
    index_128 = VectorStoreIndex(nodes=chunks_128, storage_context=storage_context)
    index_parent = VectorStoreIndex(nodes=chunks_parent, storage_context=storage_context)
    index_hierarchical = VectorStoreIndex(nodes=chunks_hierarchical, storage_context=storage_context)

    parent_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": index_parent.as_retriever()},
        node_dict=storage_context.docstore.docs,
        verbose=True,
    )
    hierarchical_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": index_hierarchical.as_retriever()},
        node_dict=storage_context.docstore.docs,
        verbose=True,
    )
    names = ["1024", "512", "256", "128", "parent", "hierarchical"]

    retrievers = [index.as_retriever() for index in [index_1024, index_512, index_256, index_128]] + [parent_retriever, hierarchical_retriever]

    retriever_dict = {name: retriever for name, retriever in zip(names, retrievers)}
    return retriever_dict


def evaluate(eval_llm: LLM, dataset_filename: str, docs: List[Document]):
    retriever_dict = get_retrievers(docs)
    # optional
    # eval_dataset = LabelledRagDataset.from_json("data/llama2_eval_qr_dataset.json")
    eval_dataset = QueryResponseDataset.from_json(dataset_filename)

    # NOTE: can uncomment other evaluators
    evaluator_c = CorrectnessEvaluator(llm=eval_llm)
    evaluator_s = SemanticSimilarityEvaluator(embed_model=Settings.embed_model)
    evaluator_r = RelevancyEvaluator(llm=eval_llm)
    evaluator_f = FaithfulnessEvaluator(llm=eval_llm)
    # pairwise_evaluator = PairwiseComparisonEvaluator(llm=eval_llm)

    eval_qs = eval_dataset.questions
    qr_pairs = eval_dataset.qr_pairs
    ref_response_strs = [r for (_, r) in qr_pairs]

    results = {}
    for name, retriever in retriever_dict.items():
        pred_responses = get_responses(eval_qs, RetrieverQueryEngine.from_args(retriever), show_progress=True)

        evaluator_dict = {
            "correctness": evaluator_c,
            "faithfulness": evaluator_f,
            "relevancy": evaluator_r,
            "semantic_similarity": evaluator_s,
        }
        batch_runner = BatchEvalRunner(evaluator_dict, workers=1, show_progress=True)

        eval_results_ = batch_runner.evaluate_responses(
            eval_qs, responses=pred_responses, reference=ref_response_strs
        )
        results[name] = eval_results_

    names = list(results.keys())
    eval_results = [results[name] for name in names]
    results_df = get_results_df(
        eval_results,
        names,
        ["correctness", "relevancy", "faithfulness", "semantic_similarity"],
    )
    print("DISPLAYING RESULTS:")
    print(results_df)
    Path("storage/finance/evaluation/llama3-70b").mkdir(parents=True, exist_ok=True)
    results_df.to_csv("storage/finance/evaluation/llama3-70b/eval_results.csv")
    results_df.to_html("storage/finance/evaluation/llama3-70b/eval_results.html")
    results_df.to_json("storage/finance/evaluation/llama3-70b/eval_results.json")



def main():
    doc_dict = get_wikipedia_finance()
    docs = list(doc_dict.values())

    generate_eval_dataset(llm=llm, nodes=docs)
    evaluate(eval_llm=llm, dataset_filename=eval_dataset_filename, docs=docs)

    # base_retriever, retriever = get_retriever(llm, docs)
    # evaluate(eval_llm=llm, embed_model=embed_model, base_retriever=base_retriever, retriever=retriever)


if __name__ == '__main__':
    nest_asyncio.apply()

    llm = OpenAI(model=DEFAULT_OPENAI_MODEL)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.callback_manager = CallbackManager()
    Settings.context_window = 16384
    main()

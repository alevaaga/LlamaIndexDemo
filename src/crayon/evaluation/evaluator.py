import asyncio
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import cast, Dict, List, Union, Type, Any

import nest_asyncio
from llama_index.core import Settings, Document
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llama_dataset import LabeledRagDataset
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai.base import OpenAI
from tonic_validate.metrics import AnswerSimilarityMetric
from llama_index.evaluation.tonic_validate import (
    AnswerConsistencyEvaluator,
    AnswerSimilarityEvaluator,
    AugmentationAccuracyEvaluator,
    AugmentationPrecisionEvaluator,
    RetrievalPrecisionEvaluator,
    TonicValidateEvaluator,
)
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator, BatchEvalRunner,
)
from llama_index.core.evaluation import (
    generate_question_context_pairs,
    EmbeddingQAFinetuneDataset, RetrieverEvaluator
)
import pandas as pd
import matplotlib.pyplot as plt

import crayon
from crayon.chatbots.finance_chatbot import create_chat_engine as create_chat_engine_full, load_indices


def load_test_data(base_dir, storage_context, nodes_per_index: int = 5) -> Dict[str, Any]:
    docstore = storage_context.docstore
    company_indices = load_indices()
    company_paths = filter(lambda p: p.is_dir(), Path(base_dir).glob("*"))
    datasets = {}
    for company_path in company_paths:
        company_name = company_path.name
        print(f"Loading documents for {company_name}.....")
        company_docs = company_indices.get(company_name, {})

        for year in company_docs:
            nodes = [docstore.get_node(node_id) for node_id in company_docs[year].index_struct.nodes_dict.keys()]
            random.shuffle(nodes)
            company_docs[year] = nodes[:nodes_per_index]
        datasets[company_name] = company_docs
    print("Documents loaded!")
    return datasets


async def create_rag_dataset(base_dir: Path, llm: BaseLLM, db_name: str, skip_missing=False) -> Dict[str, Any]:
    storage_context = crayon.get_storage_context_filesystem(db_name=db_name)
    datasets = load_test_data(base_dir, storage_context, nodes_per_index=5)

    eval_dataset_root = Path(crayon.EVALUATION_ROOT)
    datasets_root = eval_dataset_root / "datasets/rag"

    eval_datasets = {}
    for company in datasets:
        eval_datasets[company] = {}
        for year in datasets[company]:
            company_dataset_dir = datasets_root / company
            dataset_file = company_dataset_dir / f"{year}.json"
            if dataset_file.exists():
                eval_datasets[company][year] = LabeledRagDataset.from_json(str(dataset_file))
            elif skip_missing:
                print(f"Skipping missing dataset for {company} {year}")
            else:
                company_dataset_dir.mkdir(parents=True, exist_ok=True)
                nodes = datasets[company][year]
                dataset_generator = RagDatasetGenerator(
                    nodes,
                    llm=cast(LLM, llm),
                    num_questions_per_chunk=1,  # set the number of questions per nodes
                    show_progress=True,
                    workers=4
                )
                rag_dataset = await dataset_generator.agenerate_dataset_from_nodes()
                eval_datasets[company][year] = rag_dataset
                rag_dataset.save_json(str(dataset_file))

    return eval_datasets


def create_retriever_eval_dataset(base_dir: Path, llm: BaseLLM, db_name: str, skip_missing=False):
    storage_context = crayon.get_storage_context_filesystem(db_name=db_name)
    datasets = load_test_data(base_dir, storage_context, nodes_per_index=10)

    eval_dataset_root = Path(crayon.EVALUATION_ROOT)
    datasets_root = eval_dataset_root / "datasets/retriever"

    eval_datasets = {}
    for company in datasets:
        eval_datasets[company] = {}
        for year in datasets[company]:
            company_dataset_dir = datasets_root / company
            dataset_file = company_dataset_dir / f"{year}.json"
            if dataset_file.exists():
                eval_datasets[company][year] = EmbeddingQAFinetuneDataset.from_json(str(dataset_file))
            elif skip_missing:
                print(f"Skipping missing dataset for {company} {year}")
            else:
                company_dataset_dir.mkdir(parents=True, exist_ok=True)
                documents = datasets[company][year]
                qa_dataset = generate_question_context_pairs(
                    cast(List[TextNode], documents),
                    llm=cast(LLM, llm),
                    num_questions_per_chunk=1,
                )
                eval_datasets[company][year] = qa_dataset
                qa_dataset.save_json(str(dataset_file))

    return eval_datasets

def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score

async def run_rag_evaluations(llm: BaseLLM, eval_datasets: Dict[str, Dict[str, LabeledRagDataset]]):
    query_engine = create_chat_engine_full(llm=llm)
    all_results = {}
    for company in eval_datasets:
        company_scores = defaultdict(lambda: defaultdict(lambda: 0.0))
        eval_dir = Path(f"{crayon.EVALUATION_ROOT}/results/rag/{company}")
        eval_dir.mkdir(parents=True, exist_ok=True)
        filename = eval_dir / "scores.json"
        if filename.exists():
            continue

        for year in eval_datasets[company]:

            rag_dataset = eval_datasets[company][year]
            if len(rag_dataset.examples) == 0:
                continue

            faithfulness_gpt4 = FaithfulnessEvaluator(llm=llm)
            relevancy_gpt4 = RelevancyEvaluator(llm=llm)
            correctness_gpt4 = CorrectnessEvaluator(llm=llm)
            runner = BatchEvalRunner(
                {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4, "correctness": correctness_gpt4},
                workers=4,
            )

            questions = [ex.query for ex in rag_dataset.examples]
            eval_results = await runner.aevaluate_queries(
                query_engine, queries=questions
            )
            for metric in eval_results.keys():
                score = get_eval_results(metric, eval_results)
                company_scores[metric][year] = score

        all_results[company] = company_scores

        with open(filename, "w") as f:
            json.dump(company_scores, f, indent=3)


    eval_dir = Path(f"{crayon.EVALUATION_ROOT}/results/rag")
    eval_dir.mkdir(parents=True, exist_ok=True)
    filename = eval_dir / "scores.json"
    with open(filename, "w") as f:
        json.dump(all_results, f, indent=3)

    avg_values = defaultdict(lambda: defaultdict(lambda: 0.0))
    avg_metrics = defaultdict(lambda: 0.0)
    for company in all_results:
        for metric in all_results[company]:
            values = all_results[company][metric].values()
            avg = sum(values) / len(values)
            avg_values[company][metric] = avg
            avg_metrics[metric] += avg

    pd.DataFrame(avg_values).plot(kind="bar", figsize=(10, 6))
    plt.savefig(eval_dir / "avg_scores.png")

    pd.DataFrame(avg_metrics, index=["avg"]).plot(kind="bar", figsize=(10, 6))
    plt.savefig(eval_dir / "avg_metrics.png")




async def run_retrieval_evaluations(eval_datasets: Dict[str, Dict[str, EmbeddingQAFinetuneDataset]]):
    metrics = ["hit_rate", "mrr", "precision", "recall", "ap", "ndcg"]
    company_indices = load_indices()

    for company in eval_datasets:
        company_metric_dicts = []
        for year in eval_datasets[company]:
            rag_dataset = eval_datasets[company][year]
            retriever = company_indices[company][year].as_retriever(similarity_top_k=5)
            retriever_evaluator = RetrieverEvaluator.from_metric_names(
                metrics, retriever=retriever
            )

            eval_results = await retriever_evaluator.aevaluate_dataset(rag_dataset)

            for eval_result in eval_results:
                metric_dict = eval_result.metric_vals_dict
                company_metric_dicts.append(metric_dict)

        full_df = pd.DataFrame(company_metric_dicts)
        name = f"{company}_top5"
        columns = {
            "retrievers": [name],
            **{k: [full_df[k].mean()] for k in metrics},
        }
        metric_df = pd.DataFrame(columns)

        eval_dir = Path(f"{crayon.EVALUATION_ROOT}/results/retrieval/{company}")
        eval_dir.mkdir(parents=True, exist_ok=True)
        metric_df.to_json(eval_dir / f"{name}.json")

        metric_df.plot.bar(title=f"{company} Retrieval Metrics", figsize=(10, 6))
        plt.savefig(eval_dir / f"{name}.png")


def main(base_dir: Path, llm: BaseLLM, db_name: str):
    rag_eval_datasets = asyncio.get_event_loop().run_until_complete(create_rag_dataset(base_dir, llm, skip_missing=False, db_name=db_name))
    qa_eval_dataset = create_retriever_eval_dataset(base_dir, llm, db_name=db_name)

    asyncio.get_event_loop().run_until_complete(run_rag_evaluations(llm, rag_eval_datasets))
    asyncio.get_event_loop().run_until_complete(run_retrieval_evaluations(qa_eval_dataset))



if __name__ == '__main__':
    nest_asyncio.apply()
    input_dir = "data/Knowledgebase/Finance"

    # llm = OpenAI(model="gpt-4o")
    llm = OpenAI(
        api_base="http://wizzo.akhbar.home:5000/v1",
        api_key="sk-ollama",
        model="gpt-4-turbo",
        temperature=0.3,
        timeout=180,
        verbose=True
    )
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.callback_manager = CallbackManager()
    Settings.context_window = 16384
    main(base_dir=Path(input_dir), llm=llm, db_name="MultiIndex")

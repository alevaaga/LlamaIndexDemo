from typing import List, Any, Dict

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.indices.base import BaseIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TransformComponent

from crayon import get_storage_context_filesystem
from crayon.ingestion.indexing import IndexStrategy


class SimpleIndexStrategy(IndexStrategy):
    def build_index(
        self,
        llm: BaseLLM,
        embed_model: BaseEmbedding,
        namespace: str,
        nodes: List[BaseNode] | Dict[str, List[BaseNode]],
        transforms: List[TransformComponent] | None = None
    ) -> Dict[Any, BaseIndex]:
        index_set = {}

        storage_context = get_storage_context_filesystem(db_name=self.db_name)
        if transforms is not None:
            processed_nodes = IngestionPipeline(transformations=transforms).run(nodes=nodes, in_place=False, show_progress=True)
        else:
            processed_nodes = nodes

        index_set[namespace] = VectorStoreIndex(
            nodes=processed_nodes,
            embed_model=Settings.embed_model,
            storage_context=storage_context,
            callback_manager=Settings.callback_manager,
            show_progress=True,
        )
        index_set[namespace].set_index_id(f"{namespace}_{namespace}_VectorIndex")
        storage_context.persist(persist_dir=storage_context.persist_dir)
        return index_set


class IndexByYear(IndexStrategy):
    def _create_node_dict(self, nodes: List[BaseNode]) -> Dict[str, List[BaseNode]]:
        # sort the documents by year
        year_dict = {}
        for node in nodes:
            year = node.metadata["year"]
            if year not in year_dict:
                year_dict[year] = []
            year_dict[year].append(node)
        return year_dict

    def build_index(
        self,
        llm: BaseLLM,
        embed_model: BaseEmbedding,
        namespace: str,
        nodes: List[BaseNode] | Dict[str, List[BaseNode]],
        transforms: List[TransformComponent] | None = None,
    ) -> Dict[Any, BaseIndex]:
        index_set = {}
        storage_context = get_storage_context_filesystem(db_name=self.db_name)

        if isinstance(nodes, list):
            year_dict = self._create_node_dict(nodes)
        else:
            year_dict = nodes

        for year in year_dict.keys():
            year_nodes = year_dict[year]
            if transforms is not None:
                processed_nodes = IngestionPipeline(transformations=transforms).run(nodes=year_nodes, in_place=False, show_progress=True)
            else:
                processed_nodes = year_nodes

            index_set[year] = VectorStoreIndex(
                nodes=processed_nodes,
                embed_model=embed_model,
                storage_context=storage_context,
                callback_manager=Settings.callback_manager,
                show_progress=True,
            )
            index_set[year].set_index_id(f"{namespace}_{year}_VectorIndex")
        storage_context.persist(persist_dir=storage_context.persist_dir)
        return index_set


IndexStrategies = {
    "simple": SimpleIndexStrategy,
    "by_year": IndexByYear,
}

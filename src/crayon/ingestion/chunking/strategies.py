from typing import List, Dict

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, IndexNode

from crayon.ingestion.chunking import ChunkingStrategy


class SimpleChunking(ChunkingStrategy):
    debug: bool = False
    chunk_size: int = 1024
    chunk_overlap: int = 0

    def chunk(self, nodes: List[BaseNode]) -> List[BaseNode]:
        base_splitter = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        nodes = base_splitter.get_nodes_from_documents(nodes)
        if self.debug:
            for sn in nodes:
                sn.metadata["chunk_size"] = self.chunk_size
        return nodes

    @classmethod
    def create(cls, chunk_size: int, chunk_overlap: int = 0, debug: bool = False):
        return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, debug=debug)


class ParentDocumentChunking(ChunkingStrategy):
    debug: bool = False
    base_size: int = 1024
    sub_chunk_sizes: List[int] = [128, 256]
    chunk_overlap: int = 0

    def chunk(self, docs: List[BaseNode]) -> List[BaseNode]:
        base_splitter = SentenceSplitter(chunk_size=self.base_size)
        all_nodes = []
        base_nodes = base_splitter.get_nodes_from_documents(docs)
        for base_node in base_nodes:
            if self.debug:
                base_node.metadata["chunk_size"] = self.base_size

            for chunk_size in self.sub_chunk_sizes:
                n = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=self.chunk_overlap)
                sub_nodes = n.get_nodes_from_documents([base_node])
                if self.debug:
                    for sn in sub_nodes:
                        sn.metadata["chunk_size"] = chunk_size

                sub_inodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                all_nodes.extend(sub_inodes)

            parent_index_node = IndexNode.from_text_node(base_node, base_node.node_id)
            all_nodes.append(parent_index_node)

        return all_nodes

    @classmethod
    def create(cls, base_size: int, sub_chunk_sizes: List[int], chunk_overlap: int = 0, debug: bool = False):
        return cls(base_size=base_size, sub_chunk_sizes=sub_chunk_sizes, chunk_overlap=chunk_overlap, debug=debug)


ChunkingStrategies = {
    "simple": SimpleChunking,
    "parent": ParentDocumentChunking
}

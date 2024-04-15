from pathlib import Path
from typing import Optional

from llama_index.core import StorageContext
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.graph_stores.types import GraphStore
from llama_index.core.storage.docstore import SimpleDocumentStore, BaseDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.index_store.types import BaseIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStore

STORAGE_ROOT = "./storage/Finance"
CACHE_ROOT = "storage/cache"


class StorageContextWrapper(StorageContext):
    persist_dir: Optional[str] = None

    @classmethod
    def from_defaults(
            cls,
            persist_dir: Optional[str] = None,
            **kwargs
    ) -> "StorageContext":
        """Create a StorageContext from defaults.

        Args:
            docstore (Optional[BaseDocumentStore]): document store
            index_store (Optional[BaseIndexStore]): index store
            vector_store (Optional[VectorStore]): vector store
            graph_store (Optional[GraphStore]): graph store
            image_store (Optional[VectorStore]): image store

        """

        storage_context = super().from_defaults(persist_dir=persist_dir, **kwargs)
        storage_context.persist_dir = persist_dir
        return storage_context


def get_storage_context_filesystem(db_name: str):
    persist_dir = Path(STORAGE_ROOT) / db_name
    index_names = ["index_store.json", "docstore.json", "default__vector_store.json", "graph_store.json"]
    existing_files = [p.name for p in persist_dir.glob("*")] if persist_dir.exists() else []
    if persist_dir.exists() and all(ind in existing_files for ind in index_names):
        print("Loading existing storage context....")
        storage_context = StorageContextWrapper.from_defaults(
            persist_dir=str(persist_dir),
            index_store=SimpleIndexStore.from_persist_path(f"{str(persist_dir)}/index_store.json"),
            docstore=SimpleDocumentStore.from_persist_path(f"{str(persist_dir)}/docstore.json"),
            vector_store=SimpleVectorStore.from_persist_path(f"{str(persist_dir)}/default__vector_store.json"),
            graph_store=SimpleGraphStore.from_persist_path(f"{str(persist_dir)}/graph_store.json"),
        )
    else:
        print("Creating new storage context....")
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        storage_context = StorageContextWrapper.from_defaults(
            persist_dir=str(persist_dir),
            index_store=SimpleIndexStore(),
            docstore=SimpleDocumentStore(),
            graph_store=SimpleGraphStore(),
            vector_store=SimpleVectorStore(),
        )

    return storage_context

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from datasets.search import BaseIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.schema import BaseNode, TransformComponent
from pydantic import BaseModel


class IndexStrategy(ABC, BaseModel):
    db_name: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    @abstractmethod
    def build_index(self, llm: BaseLLM, embed_model: BaseEmbedding, namespace: str, nodes: List[BaseNode] | Dict[str, List[BaseNode]], transforms: List[TransformComponent] | None = None) -> Dict[Any, BaseIndex]:
        pass

    @classmethod
    def create(cls, db_name: str, **kwargs):
        return cls(db_name=db_name, **kwargs)

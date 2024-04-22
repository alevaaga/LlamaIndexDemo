from abc import ABC, abstractmethod
from typing import List

from llama_index.core.schema import BaseNode
from pydantic import BaseModel


class ChunkingStrategy(ABC, BaseModel):
    @abstractmethod
    def chunk(self, nodes: List[BaseNode]) -> List[BaseNode]:
        ...

    @classmethod
    def create(cls, **kwargs):
        return cls(**kwargs)



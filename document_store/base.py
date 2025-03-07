from typing import List
from abc import ABC, abstractmethod
import hashlib
import json

import uuid
from dataclasses import dataclass, field


# Create a custom namespace UUID for our document store
# We use uuid5 with the DNS namespace to create our own unique namespace
DOCUMENT_STORE_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_DNS,  # Standard DNS namespace
    'document.store'  # Our unique domain
)


@dataclass
class Document:
    text: str
    tags: set[str] = field(default_factory=set)
    metadata: dict[str, str] = None
    uuid: str = field(init=False)

    def __post_init__(self):
        """Initialize UUID based on text hash."""
        self.uuid = str(uuid.uuid5(DOCUMENT_STORE_NAMESPACE, self.hash()))

    def hash(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def __hash__(self):
        # use self.hash()
        return hash(self.hash())

class DocumentStore(ABC):

    def get_document_by_id(self, id: str) -> Document:
        return self.documents[id]
    
    @abstractmethod
    def add_document(self, document: Document, ignore_dublicates: bool = True) -> str:
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Document]:
        pass
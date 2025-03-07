from typing import List, Set
import numpy as np

from .basic import BasicDocumentStore, OpenAIMixin, SentenceTransformerMixin
from .base import Document
from .tags import evaluate_many


class TagDocumentStore(BasicDocumentStore):

    @property
    def all_tags(self) -> Set[str]:
        tags = set()
        for doc in self.documents.values():
            tags |= doc.tags
        return tags

    def search(self, query: str, tag_expression: str | None = None, top_k: int = 10) -> List[Document]:
        """Search for similar documents using cosine similarity"""
        if not self.documents:
            return []
            
        # Get embedding for query
        query_embedding = self._get_embedding(query)
        
        if tag_expression:
            # Filter documents by tag expression
            mask = evaluate_many(tag_expression, [doc.tags for doc in self.documents.values()])
            doc_ids = [doc_id for doc_id, m in zip(self.documents.keys(), mask) if m]
        else:
            # Vectorized cosine similarity calculation
            doc_ids = list(self.embeddings.keys())
        
        # Stack all document embeddings into a matrix
        doc_embeddings = np.vstack([self.embeddings[doc_id] for doc_id in doc_ids])
        
        # Normalize query embedding
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Normalize all document embeddings (row-wise)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
        normalized_docs = doc_embeddings / doc_norms
        
        # Calculate dot product (cosine similarity since vectors are normalized)
        similarities = np.dot(normalized_docs, query_norm)
        
        # Get indices of top-k similarities
        top_indices = np.argsort(-similarities)[:top_k]
        
        # Map back to document IDs and return documents
        top_doc_ids = [doc_ids[i] for i in top_indices]
        return [self.documents[doc_id] for doc_id in top_doc_ids]


class OpenAITagDocumentStore(OpenAIMixin, TagDocumentStore):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        OpenAIMixin.__init__(self, model_name)
        TagDocumentStore.__init__(self)

class SentenceTransformerTagDocumentStore(SentenceTransformerMixin, TagDocumentStore):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        OpenAIMixin.__init__(self, model_name)
        TagDocumentStore.__init__(self)


from typing import List, Dict
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from abc import abstractmethod
import pickle

from .base import Document, DocumentStore


class BasicDocumentStore(DocumentStore):
    def __init__(self):
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}


    @abstractmethod
    def _get_embedding(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        pass
        
    def add_document(self, document: Document, ignore_dublicates: bool = True) -> str:
        """Add document to store and compute its embedding"""
        if document.uuid in self.documents and ignore_dublicates:
            return document.uuid
            
        self.documents[document.uuid] = document
        self.embeddings[document.uuid] = self._get_embedding(document.text)
        return document.uuid

    def add_documents(self, documents: List[Document], ignore_duplicates: bool = True) -> List[str]:
        """Add multiple documents to store"""
        filtered_documents = []
        for doc in documents:
            if doc.uuid not in self.documents or not ignore_duplicates:
                filtered_documents.append(doc)
        embeddings = self._get_embeddings([doc.text for doc in filtered_documents])
        for doc, embedding in zip(filtered_documents, embeddings):
            self.documents[doc.uuid] = doc
            self.embeddings[doc.uuid] = embedding
        return [doc.uuid for doc in filtered_documents]

        
    def search(self, query: str, top_k: int = 10) -> List[Document]:
        """Search for similar documents using cosine similarity"""
        if not self.documents:
            return []
            
        # Get embedding for query
        query_embedding = self._get_embedding(query)
        
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

    def store(self, path: str):
        """Store the document store to disk"""
        with open(path, 'wb') as f:
            pickle.dump((self.documents, self.embeddings), f)

    def load(self, path: str):
        """Load the document store from disk"""
        with open(path, 'rb') as f:
            self.documents, self.embeddings = pickle.load(f)


class OpenAIMixin:
    batch_size: int = 512

    def __init__(self, model_name: str = "text-embedding-3-small", client: OpenAI | None = None):
        self.model_name = model_name
        self.client = client or OpenAI()
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from OpenAI API"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        return np.array(response.data[0].embedding)

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        embeddings = []
        for batch in range(0, len(texts), self.batch_size):
            print('!')
            batch_texts = texts[batch:batch+self.batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch_texts
            )
            embeddings.extend([res.embedding for res in response.data])
        return np.array(embeddings)


class OpenAIDocumentStore(OpenAIMixin, BasicDocumentStore):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        OpenAIMixin.__init__(self, model_name)
        BasicDocumentStore.__init__(self)


class SentenceTransformerMixin:
    batch_size: int = 32
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def _get_embedding(self, text: str | List[str]) -> np.ndarray:
        """Get embedding from Sentence Transformers model"""
        return self.model.encode(text, convert_to_numpy=True, batch_size=self.batch_size)
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embedding from Sentence Transformers model"""
        return self._get_embedding(texts)

class SentenceTransformerStore(SentenceTransformerMixin, BasicDocumentStore):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        OpenAIMixin.__init__(self, model_name)
        BasicDocumentStore.__init__(self)


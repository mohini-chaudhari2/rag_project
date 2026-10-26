"""Embedding configuration for RAG system."""
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def get_embeddings():
    """Get the embedding model for vector store."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


class EmbeddingWrapper:
    """Wrapper to make embeddings compatible with both ChromaDB and FAISS."""
    
    def __init__(self):
        self.embeddings = get_embeddings()
    
    def embed_documents(self, texts):
        """Embed a list of documents."""
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text):
        """Embed a single query."""
        return self.embeddings.embed_query(text)
    
    def __call__(self, text):
        """Make the wrapper callable for ChromaDB compatibility."""
        if isinstance(text, list):
            return self.embed_documents(text)
        return self.embed_query(text)
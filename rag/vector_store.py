"""Vector store management with ChromaDB and FAISS."""
from typing import List, Dict, Optional
import os
from pathlib import Path
import pickle

from langchain_community.vectorstores import Chroma, FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv

from rag.embeddings import EmbeddingWrapper

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


class VectorStoreManager:
    """Manage both ChromaDB and FAISS vector stores."""
    
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.embeddings = EmbeddingWrapper()
        self.chroma_store = None
        self.faiss_store = None
        self.faiss_path = os.path.join(persist_dir, "faiss_index")
        
        # Ensure persist directory exists
        os.makedirs(persist_dir, exist_ok=True)
        
        # Load existing stores
        self._load_stores()
    
    def _load_stores(self):
        """Load existing vector stores if available."""
        # Load ChromaDB
        try:
            self.chroma_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"ChromaDB not found, will create new: {e}")
            self.chroma_store = None
        
        # Load FAISS
        if os.path.exists(f"{self.faiss_path}.pkl"):
            try:
                self.faiss_store = FAISS.load_local(
                    self.faiss_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"FAISS index not found, will create new: {e}")
                self.faiss_store = None
    
    def add_documents(self, chunks: List[Dict], file_id: Optional[str] = None):
        """
        Add document chunks to both vector stores.
        
        Args:
            chunks: List of document chunks with text and metadata
            file_id: Optional file identifier for tracking
        """
        if not chunks:
            return
        
        # Convert to LangChain Documents
        documents = []
        for chunk in chunks:
            text = chunk.get("text", "")
            meta = chunk.get("meta", {})
            if file_id:
                meta["file_id"] = file_id
            documents.append(Document(page_content=text, metadata=meta))
        
        # Add to ChromaDB
        if self.chroma_store is None:
            self.chroma_store = Chroma.from_documents(
                documents,
                self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            self.chroma_store.add_documents(documents)
        
        # Persist ChromaDB
        try:
            self.chroma_store.persist()
        except AttributeError:
            # Newer versions auto-persist
            pass
        
        # Add to FAISS
        if self.faiss_store is None:
            self.faiss_store = FAISS.from_documents(documents, self.embeddings)
        else:
            self.faiss_store.add_documents(documents)
        
        # Save FAISS index
        self.faiss_store.save_local(self.faiss_path)
    
    def similarity_search(self, query: str, k: int = 5, 
                         use_faiss: bool = True,
                         filter_dict: Optional[Dict] = None) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            use_faiss: Whether to use FAISS (faster) or ChromaDB
            filter_dict: Optional metadata filter
            
        Returns:
            List of relevant documents
        """
        if use_faiss and self.faiss_store:
            # FAISS doesn't support filtering, so we search and filter manually
            results = self.faiss_store.similarity_search(query, k=k*2)
            
            if filter_dict:
                filtered = []
                for doc in results:
                    if all(doc.metadata.get(k) == v for k, v in filter_dict.items()):
                        filtered.append(doc)
                        if len(filtered) >= k:
                            break
                return filtered[:k]
            
            return results[:k]
        
        elif self.chroma_store:
            # ChromaDB supports native filtering
            if filter_dict:
                return self.chroma_store.similarity_search(
                    query, k=k, filter=filter_dict
                )
            return self.chroma_store.similarity_search(query, k=k)
        
        return []
    
    def get_all_file_ids(self) -> List[str]:
        """Get list of all unique file IDs in the store."""
        if not self.chroma_store:
            return []
        
        try:
            # Get all documents and extract unique file_ids
            results = self.chroma_store.get()
            file_ids = set()
            
            if results and "metadatas" in results:
                for meta in results["metadatas"]:
                    if meta and "file_id" in meta:
                        file_ids.add(meta["file_id"])
            
            return list(file_ids)
        except Exception as e:
            print(f"Error getting file IDs: {e}")
            return []
    
    def delete_by_file_id(self, file_id: str):
        """
        Delete all chunks associated with a file ID.
        
        Note: This is complex with FAISS, so we rebuild it.
        """
        if not self.chroma_store:
            return
        
        try:
            # Delete from ChromaDB
            results = self.chroma_store.get(where={"file_id": file_id})
            if results and "ids" in results:
                self.chroma_store.delete(ids=results["ids"])
                
                # Persist changes
                try:
                    self.chroma_store.persist()
                except AttributeError:
                    pass
            
            # Rebuild FAISS without the deleted file
            remaining_docs = self.chroma_store.get()
            if remaining_docs and "documents" in remaining_docs:
                docs = []
                metas = remaining_docs.get("metadatas", [])
                for i, text in enumerate(remaining_docs["documents"]):
                    meta = metas[i] if i < len(metas) else {}
                    docs.append(Document(page_content=text, metadata=meta))
                
                if docs:
                    self.faiss_store = FAISS.from_documents(docs, self.embeddings)
                    self.faiss_store.save_local(self.faiss_path)
                else:
                    self.faiss_store = None
        except Exception as e:
            print(f"Error deleting file {file_id}: {e}")
    
    def clear_all(self):
        """Clear all vector stores."""
        # Clear ChromaDB
        if os.path.exists(self.persist_dir):
            import shutil
            shutil.rmtree(self.persist_dir)
            os.makedirs(self.persist_dir, exist_ok=True)
        
        self.chroma_store = None
        self.faiss_store = None
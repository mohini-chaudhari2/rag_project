"""Text chunking strategies for document processing."""
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))


def chunk_documents(docs: List[Dict], chunk_size: int = CHUNK_SIZE, 
                   chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        docs: List of document dicts with 'text' and 'meta' keys
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked documents with metadata
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_docs = []
    
    for doc in docs:
        text = doc.get("text", "")
        meta = doc.get("meta", {})
        
        if not text or not text.strip():
            continue
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Create new documents for each chunk
        for i, chunk in enumerate(chunks):
            chunk_meta = meta.copy()
            chunk_meta["chunk_id"] = i
            chunk_meta["total_chunks"] = len(chunks)
            
            chunked_docs.append({
                "text": chunk,
                "meta": chunk_meta
            })
    
    return chunked_docs


def smart_chunk_by_type(docs: List[Dict]) -> List[Dict]:
    """
    Apply different chunking strategies based on document type.
    
    Args:
        docs: List of document dicts
        
    Returns:
        List of appropriately chunked documents
    """
    result = []
    
    for doc in docs:
        doc_type = doc.get("meta", {}).get("type", "txt")
        
        if doc_type == "csv":
            # CSV files are already structured, use smaller chunks
            result.extend(chunk_documents([doc], chunk_size=500, chunk_overlap=50))
        elif doc_type == "pdf":
            # PDFs may have complex formatting, use standard chunks
            result.extend(chunk_documents([doc], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP))
        else:
            # Default chunking for other types
            result.extend(chunk_documents([doc]))
    
    return result
"""Document processing pipeline."""
from typing import List, Dict, Optional
import os
from pathlib import Path

from processing.file_loaders.pdf_loader import load_pdf
from processing.file_loaders.docx_loader import load_docx
from processing.file_loaders.txt_loader import load_txt
from processing.file_loaders.md_loader import load_md
from processing.file_loaders.csv_loader import load_csv
from processing.chunker import smart_chunk_by_type


class DocumentProcessor:
    """Process and load various document types."""
    
    LOADERS = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".doc": load_docx,
        ".txt": load_txt,
        ".md": load_md,
        ".markdown": load_md,
        ".csv": load_csv,
    }
    
    @classmethod
    def load_file(cls, file_path: str, file_id: Optional[str] = None) -> List[Dict]:
        """
        Load a single file and return processed documents.
        
        Args:
            file_path: Path to the file
            file_id: Optional unique identifier for the file
            
        Returns:
            List of document chunks with metadata
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        
        if extension not in cls.LOADERS:
            raise ValueError(f"Unsupported file type: {extension}")
        
        loader = cls.LOADERS[extension]
        
        try:
            docs = loader(str(path))
        except Exception as e:
            raise RuntimeError(f"Failed to load {file_path}: {e}")
        
        # Add file metadata
        for doc in docs:
            if "meta" not in doc:
                doc["meta"] = {}
            doc["meta"]["source"] = path.name
            doc["meta"]["file_path"] = str(path)
            doc["meta"]["file_type"] = extension
            if file_id:
                doc["meta"]["file_id"] = file_id
        
        return docs
    
    @classmethod
    def process_file(cls, file_path: str, file_id: Optional[str] = None) -> List[Dict]:
        """
        Load and chunk a file.
        
        Args:
            file_path: Path to the file
            file_id: Optional unique identifier
            
        Returns:
            List of chunked documents ready for vector store
        """
        docs = cls.load_file(file_path, file_id)
        chunked = smart_chunk_by_type(docs)
        return chunked
    
    @classmethod
    def process_multiple_files(cls, file_paths: List[str], 
                              file_ids: Optional[List[str]] = None) -> List[Dict]:
        """
        Process multiple files.
        
        Args:
            file_paths: List of file paths
            file_ids: Optional list of file IDs (must match length of file_paths)
            
        Returns:
            Combined list of chunked documents
        """
        if file_ids and len(file_ids) != len(file_paths):
            raise ValueError("file_ids length must match file_paths length")
        
        all_chunks = []
        
        for i, path in enumerate(file_paths):
            fid = file_ids[i] if file_ids else None
            try:
                chunks = cls.process_file(path, fid)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Warning: Failed to process {path}: {e}")
                continue
        
        return all_chunks
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls.LOADERS.keys())


def process_uploaded_file(uploaded_file, upload_dir: str = "uploads") -> tuple:
    """
    Save and process an uploaded Streamlit file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        upload_dir: Directory to save uploaded files
        
    Returns:
        Tuple of (file_path, file_id, chunks)
    """
    os.makedirs(upload_dir, exist_ok=True)
    
    # Generate file ID
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # Save file
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process file
    chunks = DocumentProcessor.process_file(file_path, file_id)
    
    return file_path, file_id, chunks
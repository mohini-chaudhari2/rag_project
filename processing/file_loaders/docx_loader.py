# processing/file_loaders/docx_loader.py
from docx import Document
from typing import List, Dict

def load_docx(path: str) -> List[Dict]:
    docs = []
    try:
        document = Document(path)
    except Exception as e:
        raise RuntimeError(f"Could not open DOCX {path}: {e}")
    paras = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    if paras:
        combined = "\n\n".join(paras)
        docs.append({"text": combined, "meta": {"type": "docx"}})
    return docs

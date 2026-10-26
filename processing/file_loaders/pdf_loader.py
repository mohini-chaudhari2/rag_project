# processing/file_loaders/pdf_loader.py
import fitz  # PyMuPDF
from typing import List, Dict

def load_pdf(path: str) -> List[Dict]:
    docs = []
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise RuntimeError(f"Could not open PDF {path}: {e}")
    for page_no in range(len(doc)):
        page = doc.load_page(page_no)
        text = page.get_text("text")
        if text and text.strip():
            docs.append({"text": text, "meta": {"page": page_no + 1}})
    doc.close()
    return docs

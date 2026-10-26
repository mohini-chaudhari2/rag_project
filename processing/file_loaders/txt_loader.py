# processing/file_loaders/txt_loader.py
from typing import List, Dict

def load_txt(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    if text and text.strip():
        return [{"text": text, "meta": {"type": "txt"}}]
    return []

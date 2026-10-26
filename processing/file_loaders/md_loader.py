# processing/file_loaders/md_loader.py
from processing.file_loaders.txt_loader import load_txt

def load_md(path: str):
    return load_txt(path)

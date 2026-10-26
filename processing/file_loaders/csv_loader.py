# processing/file_loaders/csv_loader.py
import pandas as pd
from typing import List, Dict

def load_csv(path: str) -> List[Dict]:
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as e:
        raise RuntimeError(f"Could not read CSV {path}: {e}")
    if df.empty:
        return []
    rows_texts = []
    for _, row in df.iterrows():
        parts = [f"{col}: {row[col]}" for col in df.columns]
        rows_texts.append(" | ".join(parts))
    combined = "\n".join(rows_texts)
    return [{"text": combined, "meta": {"type": "csv", "rows": len(df)}}]

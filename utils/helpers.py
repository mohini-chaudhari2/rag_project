"""Utility helper functions."""
import hashlib
import json
from typing import List, Dict
from datetime import datetime


def generate_file_id(filename: str, size: int) -> str:
    """Generate a unique file ID based on name and size."""
    return hashlib.md5(f"{filename}_{size}".encode()).hexdigest()


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def format_timestamp(timestamp: float = None) -> str:
    """Format timestamp as readable string."""
    if timestamp is None:
        timestamp = datetime.now().timestamp()
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to max length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def extract_file_info(file) -> Dict:
    """Extract information from uploaded file."""
    return {
        "name": file.name,
        "size": file.size,
        "type": file.type,
        "size_formatted": format_file_size(file.size)
    }


def create_chat_message(role: str, content: str, 
                       sources: List[Dict] = None) -> Dict:
    """Create a standardized chat message."""
    msg = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().timestamp()
    }
    if sources:
        msg["sources"] = sources
    return msg


def save_chat_history(history: List[Dict], filepath: str):
    """Save chat history to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_chat_history(filepath: str) -> List[Dict]:
    """Load chat history from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []
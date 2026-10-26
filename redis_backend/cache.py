# redis_backend/cache.py
import json
import hashlib
from typing import Optional
from redis_backend.redis_client import get_redis_client

r = get_redis_client()

CACHE_TTL_SECONDS = 60 * 60 * 24

def _cache_key_for_query(query: str, top_k: int, temperature: float, file_ids: list, prompt_style: str):
    payload = {
        "q": query,
        "k": top_k,
        "t": float(temperature),
        "files": sorted(file_ids or []),
        "style": prompt_style  # Added: Include prompt_style in the key for style-specific caching
    }
    j = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(j.encode("utf-8")).hexdigest()
    return f"rag_cache:{h}"

def get_cached_answer(query: str, top_k: int, temperature: float, file_ids: list, prompt_style: str) -> Optional[dict]:
    key = _cache_key_for_query(query, top_k, temperature, file_ids, prompt_style)
    data = r.get(key)
    if not data:
        return None
    try:
        return json.loads(data)
    except Exception:
        return None

def set_cached_answer(query: str, top_k: int, temperature: float, file_ids: list, prompt_style: str, answer_obj: dict, ttl: int = CACHE_TTL_SECONDS):
    key = _cache_key_for_query(query, top_k, temperature, file_ids, prompt_style)
    r.set(key, json.dumps(answer_obj, ensure_ascii=False), ex=ttl)
    return key
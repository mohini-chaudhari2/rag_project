# redis_backend/redis_client.py
import os
from dotenv import load_dotenv
import redis
import sys

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

def get_redis_client():

    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, socket_connect_timeout=5, decode_responses=True)
        if not r.ping():
            raise ConnectionError("Ping failed")
        return r
    except Exception as e:
        print(f"[ERROR] Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT} — {e}")
        sys.exit(1)

if __name__ == "__main__":
    rc = get_redis_client()
    print("Redis ready:", rc.ping())

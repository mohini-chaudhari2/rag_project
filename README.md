# RAG Document Assistant

A production-ready RAG (Retrieval-Augmented Generation) application, built with LangChain, Groq API, ChromaDB, FAISS, Redis caching, and Streamlit.

## Features

- **Multi-format Document Support**: PDF, DOCX, TXT, Markdown, CSV
- **Hybrid Vector Store**: ChromaDB for persistence + FAISS for fast search
- **Intelligent Caching**: Redis-based caching for repeated queries
- **Streaming Responses**: Real-time answer streaming from Groq LLM
- **Source Attribution**: Automatic source citation with page numbers
- **UI**: Streamlit
- **Smart Chunking**: Adaptive text chunking based on document type
- **Chat History**: Persistent conversation history
- **Document Management**: Easy upload, view, and delete documents

## Architecture

```
User Query → Cache Check → Vector Search (FAISS/ChromaDB) → 
LLM (Groq) → Streaming Response → Cache Store → UI Display
```

### Tech Stack

- **LLM**: Groq API (llama-3.1-70b-versatile)
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Vector Stores**: ChromaDB + FAISS
- **Caching**: Redis
- **Framework**: LangChain
- **UI**: Streamlit
- **Document Processing**: PyMuPDF, python-docx, pandas

## Prerequisites

- Python 3.8+
- Redis server
- Groq API key ([Get one here](https://console.groq.com))

## Quick Start

### 1. Clone and Setup

```bash
# Clone the project directory

git clone https://github.com/ammarlouah/rag_project.git
cd rag_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Install and Start Redis

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**macOS:**
```bash
brew install redis
brew services start redis
```

**Windows:**
Download from [Redis for Windows](https://github.com/microsoftarchive/redis/releases)

**Verify Redis is running:**
```bash
redis-cli ping
# Should return: PONG
```

### 3. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Groq API key
nano .env  # or use any text editor
```

Required environment variables:
```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-70b-versatile
```

### 4. Source Files

- `app.py`
- `requirements.txt`
- `.env.example`
- `rag/groq_llm.py`
- `rag/embeddings.py`
- `rag/vector_store.py`
- `rag/chain.py`
- `processing/document_processor.py`
- `processing/chunker.py`
- `processing/file_loaders/*.py`
- `redis_backend/redis_client.py`
- `redis_backend/cache.py`
- `utils/helpers.py`

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Guide

### Uploading Documents

1. Click the file uploader in the sidebar
2. Select one or multiple documents (PDF, DOCX, TXT, MD, CSV)
3. Wait for processing to complete
4. Documents are automatically indexed

### Asking Questions

1. Choose the type of answering
2. Type your question in the chat input
3. Press Enter or click Send
4. Watch the AI stream its response in real-time
5. View source citations by expanding the "Sources" section

### Settings

- **Documents to retrieve**: Control how many chunks to use (1-10)
- **Response creativity**: Adjust temperature (0.0-1.0)
  - 0.0 = Focused and deterministic
  - 1.0 = Creative and varied
- **Enable Redis caching**: Toggle response caching

### Managing Documents

- **View documents**: Expand file cards in the sidebar
- **Remove documents**: Click the 🗑️ button next to any file
- **Clear chat**: Remove conversation history
- **Clear DB**: Delete all documents and reset

## Configuration

### Groq Models

Available models (set in `.env`):
- `llama-3.1-70b-versatile` (recommended for quality)
- `llama-3.1-8b-instant` (faster, lower cost)
- `mixtral-8x7b-32768` (good balance)

### Embedding Models

Change in `.env`:
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Fast, 384 dimensions
# Or alternatives:
# EMBEDDING_MODEL=all-mpnet-base-v2  # Better quality, 768 dimensions
```

### Chunking Parameters

Adjust in `.env`:
```bash
CHUNK_SIZE=1000          # Characters per chunk
CHUNK_OVERLAP=200        # Overlap between chunks
TOP_K=5                  # Default documents to retrieve
```

### Redis Configuration

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
```

## 🔧 Troubleshooting

### Redis Connection Error

```bash
# Check if Redis is running
redis-cli ping

# Start Redis
# Ubuntu/Debian:
sudo systemctl start redis-server

# macOS:
brew services start redis
```

### Groq API Error

- Verify your API key in `.env`
- Check rate limits at [Groq Console](https://console.groq.com)
- Try a different model

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### ChromaDB Errors

```bash
# Clear and rebuild
rm -rf chroma_db
# Restart the app
```

### Add New File Loaders

Create a new loader in `processing/file_loaders/`:

```python
def load_custom(path: str) -> List[Dict]:
    # Your loading logic
    return [{"text": content, "meta": {"type": "custom"}}]
```

Register in `DocumentProcessor.LOADERS`:

```python
LOADERS = {
    ".custom": load_custom,
    # ... existing loaders
}
```

## Contributing

Contributions welcome! Areas for improvement:

- Multi-modal support (images, audio)
- Advanced conversation memory
- Query rewriting and expansion
- Hybrid search (keyword + semantic)
- Document summarization pipelines

## License

MIT License - feel free to use in your projects!

## Support

For issues and questions:
- Check the troubleshooting section
- Review error logs in the terminal
- Ensure all dependencies are installed correctly

---

**Happy Document Chatting!**
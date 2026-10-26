"""Main Streamlit application for RAG NotebookLM clone."""
import streamlit as st
import os
from datetime import datetime
from typing import List, Dict

from rag.vector_store import VectorStoreManager
from rag.chain import RAGChain
from processing.document_processor import process_uploaded_file, DocumentProcessor
from redis_backend.cache import get_cached_answer, set_cached_answer
from utils.helpers import (
    extract_file_info, create_chat_message, 
    format_file_size, truncate_text
)


# Page configuration
st.set_page_config(
    page_title="NotebookLM Clone - RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for NotebookLM-like styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f1f1f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5f6368;
        margin-bottom: 2rem;
    }
    .file-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #4285f4;
    }
    .source-badge {
        background: #e8f0fe;
        color: #1967d2;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        display: inline-block;
        margin: 0.25rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStoreManager()

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = RAGChain(st.session_state.vector_store)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files_info" not in st.session_state:
    st.session_state.uploaded_files_info = {}

if "current_query" not in st.session_state:
    st.session_state.current_query = ""


def display_header():
    """Display the main header."""
    st.markdown('<div class="main-header">📚 NotebookLM Clone</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered document analysis with RAG</div>', unsafe_allow_html=True)


def display_file_upload_section():
    """Display file upload section in sidebar."""
    st.sidebar.title("📁 Document Management")
    
    # File uploader
    uploaded_files = st.sidebar.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "doc", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, TXT, MD, CSV"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_id = f"{uploaded_file.name}_{uploaded_file.size}"
            
            # Skip if already processed
            if file_id in st.session_state.uploaded_files_info:
                continue
            
            with st.sidebar.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
                try:
                    # Process file
                    file_path, fid, chunks = process_uploaded_file(uploaded_file)
                    
                    # Add to vector store
                    st.session_state.vector_store.add_documents(chunks, fid)
                    
                    # Store file info
                    st.session_state.uploaded_files_info[file_id] = {
                        "name": uploaded_file.name,
                        "size": uploaded_file.size,
                        "chunks": len(chunks),
                        "file_id": fid,
                        "timestamp": datetime.now().timestamp()
                    }
                    
                    status.update(
                        label=f"✅ {uploaded_file.name} processed successfully!",
                        state="complete"
                    )
                    
                except Exception as e:
                    status.update(
                        label=f"❌ Error processing {uploaded_file.name}",
                        state="error"
                    )
                    st.sidebar.error(f"Error: {str(e)}")


def display_uploaded_files():
    """Display list of uploaded files."""
    if not st.session_state.uploaded_files_info:
        st.sidebar.info("No documents uploaded yet. Upload documents to get started!")
        return
    
    st.sidebar.subheader("📄 Uploaded Documents")
    
    for file_id, info in st.session_state.uploaded_files_info.items():
        with st.sidebar.expander(f"📄 {info['name']}", expanded=False):
            st.write(f"**Size:** {format_file_size(info['size'])}")
            st.write(f"**Chunks:** {info['chunks']}")
            
            if st.button(f"🗑️ Remove", key=f"remove_{file_id}"):
                # Remove from vector store
                st.session_state.vector_store.delete_by_file_id(info['file_id'])
                # Remove from session
                del st.session_state.uploaded_files_info[file_id]
                st.rerun()


def display_settings():
    """Display settings section."""
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Settings")
    
    # Prompt style selection
    prompt_style = st.sidebar.selectbox(
        "Response style",
        options=["smart", "conversational", "professional"],
        help="Smart: Balanced, Conversational: Casual, Professional: Formal"
    )
    
    # Update prompt style if changed
    if hasattr(st.session_state, 'rag_chain'):
        if not hasattr(st.session_state.rag_chain, 'prompt_style') or \
           st.session_state.rag_chain.prompt_style != prompt_style:
            st.session_state.rag_chain.set_prompt_style(prompt_style)
    
    # Retrieval settings
    top_k = st.sidebar.slider(
        "Documents to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks to use for answering"
    )
    
    temperature = st.sidebar.slider(
        "Response creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Higher values = more creative, Lower values = more focused"
    )
    
    use_cache = st.sidebar.checkbox(
        "Enable Redis caching",
        value=True,
        help="Cache responses for faster repeated queries"
    )
    
    return {
        "top_k": top_k, 
        "temperature": temperature, 
        "use_cache": use_cache,
        "prompt_style": prompt_style
    }


def display_chat_interface(settings: Dict):
    """Display the main chat interface."""
    
    # Display chat history
    for message in st.session_state.chat_history:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        else:
            with st.chat_message("assistant"):
                st.write(content)
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources", expanded=False):
                        for source in message["sources"]:
                            source_text = source["source"]
                            if source.get("page"):
                                source_text += f" (Page {source['page']})"
                            st.markdown(f'<span class="source-badge">{source_text}</span>', 
                                      unsafe_allow_html=True)
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.uploaded_files_info:
            st.warning("⚠️ Please upload some documents first!")
            return
        
        # Add user message to history
        user_msg = create_chat_message("user", prompt)
        st.session_state.chat_history.append(user_msg)
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get file IDs for caching
        file_ids = [info["file_id"] for info in st.session_state.uploaded_files_info.values()]
        
        # Check cache
        cached_response = None
        if settings["use_cache"]:
            cached_response = get_cached_answer(
                prompt,
                settings["top_k"],
                settings["temperature"],
                file_ids,
                settings["prompt_style"]  # Added: Pass prompt_style for style-specific caching
            )
        
        # Display assistant response
        with st.chat_message("assistant"):
            if cached_response:
                # Use cached response
                st.write(cached_response["answer"])
                sources = cached_response.get("sources", [])
                
                # Add to history
                assistant_msg = create_chat_message("assistant", cached_response["answer"], sources)
                st.session_state.chat_history.append(assistant_msg)
                
                # Display sources
                if sources:
                    with st.expander("📚 Sources (from cache)", expanded=False):
                        for source in sources:
                            source_text = source["source"]
                            if source.get("page"):
                                source_text += f" (Page {source['page']})"
                            st.markdown(f'<span class="source-badge">{source_text}</span>', 
                                        unsafe_allow_html=True)
            else:
                # Generate new response
                response_placeholder = st.empty()
                full_response = ""
                
                # Update chain temperature
                st.session_state.rag_chain.llm.temperature = settings["temperature"]
                
                # Query RAG system
                result = st.session_state.rag_chain.query(
                    prompt,
                    top_k=settings["top_k"],
                    file_ids=file_ids,
                    stream=True
                )
                
                # Stream response
                if "answer_stream" in result:
                    for chunk in result["answer_stream"]:
                        full_response += chunk
                        response_placeholder.write(full_response + "▌")
                    response_placeholder.write(full_response)
                else:
                    full_response = result.get("answer", "No response generated.")
                    response_placeholder.write(full_response)
                
                # Get sources
                sources = result.get("sources", [])
                
                # Add to history
                assistant_msg = create_chat_message("assistant", full_response, sources)
                st.session_state.chat_history.append(assistant_msg)
                
                # Cache the response
                if settings["use_cache"]:
                    try:
                        set_cached_answer(
                            prompt,
                            settings["top_k"],
                            settings["temperature"],
                            file_ids,
                            settings["prompt_style"],  # Added: Pass prompt_style for style-specific caching
                            {"answer": full_response, "sources": sources}
                        )
                    except Exception as e:
                        st.sidebar.warning(f"Could not cache response: {e}")
                
                # Display sources
                if sources:
                    with st.expander("📚 Sources", expanded=False):
                        for source in sources:
                            source_text = source["source"]
                            if source.get("page"):
                                source_text += f" (Page {source['page']})"
                            st.markdown(f'<span class="source-badge">{source_text}</span>', 
                                      unsafe_allow_html=True)


def display_clear_buttons():
    """Display clear chat and clear database buttons."""
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        if st.button("🗄️ Clear DB"):
            st.session_state.vector_store.clear_all()
            st.session_state.uploaded_files_info = {}
            st.session_state.chat_history = []
            st.rerun()


def main():
    """Main application function."""
    display_header()
    
    # Sidebar
    display_file_upload_section()
    display_uploaded_files()
    settings = display_settings()
    display_clear_buttons()
    
    # Main chat interface
    if st.session_state.uploaded_files_info:
        display_chat_interface(settings)
    else:
        st.info("👈 Upload documents in the sidebar to get started!")
        
        # Display example queries
        st.subheader("💡 Example Use Cases")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Research & Analysis**
            - Summarize multiple research papers
            - Compare findings across documents
            - Extract key insights
            """)
        
        with col2:
            st.markdown("""
            **Document QA**
            - Ask questions about contracts
            - Find specific information in reports
            - Analyze meeting notes
            """)


if __name__ == "__main__":
    main()
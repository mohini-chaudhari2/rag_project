"""RAG chain implementation using LangChain."""
from typing import List, Dict, Optional, Generator
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from rag.groq_llm import GroqLLM
from rag.vector_store import VectorStoreManager

# Smart prompt that handles both greetings and document queries
RAG_PROMPT_TEMPLATE = """You are a knowledgeable and friendly AI assistant with access to the user's documents. 

Your role:
- Prioritize answering questions using the provided document context whenever relevant
- Respond naturally to greetings, casual conversation, and general questions unrelated to documents
- Be conversational and helpful in all interactions

Available Document Context:
{context}

User Message: {question}

Instructions:
- First, determine if the question relates to the document content. If yes, provide a detailed answer using ONLY the context above
- If the question is a greeting (hi, hello, how are you, etc.), respond warmly and offer to help with their documents
- If this is a general question unrelated to documents, answer naturally and helpfully
- Always cite which document you're referencing when answering document questions
- If you cannot find relevant information in the context for a document-related question, say: "I don't see that information in your uploaded documents. Could you rephrase your question or upload additional documents?"
- Be concise but comprehensive

Response:"""


# Conversational prompt - more casual
CONVERSATIONAL_PROMPT = """Hey! I'm your AI assistant here to help you explore your documents. Let's chat!

What's in your documents:
{context}

You asked: {question}

I'll help you out! First, if this is about your docs, I'll use only what's provided above to give you a solid answer and cite the sources. If it's just a greeting or something else, I'm happy to chat casually. If I can't find the info in the docs, I'll let you know: "I don't see that information in your uploaded documents. Could you rephrase your question or upload additional documents?" Let me know if I should look deeper or if you need clarification on anything!"""


# Professional prompt - formal tone
PROFESSIONAL_PROMPT = """I am an AI document analyst providing precise information from your uploaded materials.

Document Context:
{context}

Query: {question}

Analysis:
I will prioritize providing a thorough response based on the available documentation whenever the query relates to document content. For such questions, I will use only the provided context, cite sources, and provide detailed, accurate information. If the requested information is not available in the provided context, I will clearly state this limitation: "I don't see that information in your uploaded documents. Could you rephrase your question or upload additional documents?" For general inquiries or greetings unrelated to documents, I will respond appropriately while remaining professional."""

class RAGChain:
    """RAG chain for question answering over documents."""
    
    def __init__(self, vector_store_manager: VectorStoreManager,
                 temperature: float = 0.1,
                 prompt_style: str = "smart"):
        """
        Initialize RAG chain.
        
        Args:
            vector_store_manager: Vector store for document retrieval
            temperature: LLM temperature (0.0-1.0)
            prompt_style: "smart", "conversational", or "professional"
        """
        self.vector_store = vector_store_manager
        self.llm = GroqLLM(temperature=temperature)
        
        # Select prompt template based on style
        if prompt_style == "conversational":
            template = CONVERSATIONAL_PROMPT
        elif prompt_style == "professional":
            template = PROFESSIONAL_PROMPT
        else:
            template = RAG_PROMPT_TEMPLATE
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        self.prompt_style = prompt_style
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents as context."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page})" if page else ""
            
            formatted.append(f"[Document {i} - {source}{page_info}]\n{doc.page_content}\n")
        
        return "\n".join(formatted)
    
    def query(self, question: str, top_k: int = 5, 
             file_ids: Optional[List[str]] = None,
             stream: bool = True) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            file_ids: Optional list of file IDs to filter by
            stream: Whether to stream the response
            
        Returns:
            Dict with answer, sources, and metadata
        """
        # Build filter if file_ids provided
        filter_dict = None
        if file_ids and len(file_ids) == 1:
            filter_dict = {"file_id": file_ids[0]}
        
        # Retrieve relevant documents
        docs = self.vector_store.similarity_search(
            question, 
            k=top_k,
            filter_dict=filter_dict
        )
        
        if not docs:
            # No documents found - likely a greeting or general question
            return {
                "answer_stream": self._handle_no_context(question, stream),
                "sources": [],
                "retrieved_docs": 0
            } if stream else {
                "answer": self._handle_no_context_sync(question),
                "sources": [],
                "retrieved_docs": 0
            }
        
        # Format context
        context = self._format_docs(docs)
        
        # Generate prompt
        prompt = self.prompt.format(context=context, question=question)
        
        # Get answer from LLM
        if stream:
            return {
                "answer_stream": self._stream_answer(prompt),
                "sources": self._extract_sources(docs),
                "retrieved_docs": len(docs),
                "documents": docs
            }
        else:
            answer = self.llm._call(prompt)
            return {
                "answer": answer,
                "sources": self._extract_sources(docs),
                "retrieved_docs": len(docs),
                "documents": docs
            }
    
    def _handle_no_context(self, question: str, stream: bool) -> Generator[str, None, None]:
        """Handle questions when no documents are found (likely greetings)."""
        # Check if it's a greeting
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        q_lower = question.lower()
        
        if any(greeting in q_lower for greeting in greetings):
            response = "Hello! 👋 I'm your AI document assistant. I can help you analyze and get insights from your uploaded documents. Feel free to ask me anything about your files, or just chat! How can I help you today?"
        else:
            response = "I'd be happy to help! However, I don't have any documents to reference for this question. You can upload documents using the sidebar, and then I'll be able to answer questions about them. Is there anything else I can help you with?"
        
        if stream:
            for word in response.split():
                yield word + " "
        else:
            yield response
    
    def _handle_no_context_sync(self, question: str) -> str:
        """Synchronous version of no context handler."""
        return "".join(self._handle_no_context(question, stream=False))
    
    def _stream_answer(self, prompt: str) -> Generator[str, None, None]:
        """Stream answer from LLM."""
        for chunk in self.llm._stream(prompt):
            yield chunk
    
    def _extract_sources(self, docs: List[Document]) -> List[Dict]:
        """Extract source information from documents."""
        sources = []
        seen = set()
        
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page")
            
            # Create unique key
            key = f"{source}_{page}" if page else source
            
            if key not in seen:
                seen.add(key)
                source_info = {
                    "source": source,
                    "page": page,
                    "file_type": doc.metadata.get("file_type", ""),
                }
                sources.append(source_info)
        
        return sources
    
    def set_prompt_style(self, style: str):
        """Change the prompt style on the fly."""
        if style == "conversational":
            template = CONVERSATIONAL_PROMPT
        elif style == "professional":
            template = PROFESSIONAL_PROMPT
        else:
            template = RAG_PROMPT_TEMPLATE
        
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        self.prompt_style = style


def create_rag_chain(vector_store_manager: VectorStoreManager,
                    temperature: float = 0.1,
                    prompt_style: str = "smart") -> RAGChain:
    """Factory function to create RAG chain."""
    return RAGChain(vector_store_manager, temperature, prompt_style)
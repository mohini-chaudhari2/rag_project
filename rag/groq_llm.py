# rag/groq_llm.py
from typing import Optional, List, Mapping, Any
from langchain_core.language_models import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from dotenv import load_dotenv
import os

load_dotenv()

try:
    from groq import Groq
except Exception as e:
    raise ImportError("The `groq` package is required for GroqLLM. Install with `pip install groq`.") from e

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", 1024))


class GroqLLM(LLM):
    """Custom LLM wrapper for Groq API."""
    
    api_key: Optional[str] = GROQ_API_KEY
    model: str = GROQ_MODEL
    temperature: float = 0.0
    max_tokens: int = GROQ_MAX_TOKENS
    top_p: float = 1.0
    client: Optional[Any] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize Groq client
        if self.api_key:
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = Groq()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "groq"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the Groq API."""
        if not self.client:
            raise RuntimeError("Groq client is not initialized.")

        messages = [{"role": "user", "content": prompt}]

        try:
            # Try with max_completion_tokens first
            completion_iterable = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=float(self.temperature),
                max_completion_tokens=int(self.max_tokens),
                top_p=float(self.top_p),
                stream=True,
                stop=stop or []
            )
        except TypeError:
            # Fallback to max_tokens
            completion_iterable = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=float(self.temperature),
                max_tokens=int(self.max_tokens),
                top_p=float(self.top_p),
                stream=True,
                stop=stop or []
            )

        # Collect all chunks
        output_chunks = []
        try:
            for chunk in completion_iterable:
                try:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None) if delta else None
                    
                    if not content:
                        try:
                            content = chunk.choices[0].message.content
                        except Exception:
                            content = None
                    
                    if content:
                        output_chunks.append(content)
                except Exception:
                    s = str(chunk)
                    if s:
                        output_chunks.append(s)
        except Exception as e:
            # Fallback to non-streaming
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=float(self.temperature),
                    max_tokens=int(self.max_tokens),
                    top_p=float(self.top_p),
                    stream=False,
                    stop=stop or []
                )
                
                if hasattr(resp, 'choices') and len(resp.choices) > 0:
                    content = resp.choices[0].message.content
                    if content:
                        return content
                
                return str(resp)
            except Exception as fallback_error:
                raise RuntimeError(f"Groq API call failed: {e}. Fallback also failed: {fallback_error}")

        return "".join(output_chunks)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Stream the output from Groq API."""
        if not self.client:
            raise RuntimeError("Groq client is not initialized.")

        messages = [{"role": "user", "content": prompt}]

        try:
            # Try with max_completion_tokens first
            completion_iterable = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=float(self.temperature),
                max_completion_tokens=int(self.max_tokens),
                top_p=float(self.top_p),
                stream=True,
                stop=stop or []
            )
        except TypeError:
            # Fallback to max_tokens
            completion_iterable = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=float(self.temperature),
                max_tokens=int(self.max_tokens),
                top_p=float(self.top_p),
                stream=True,
                stop=stop or []
            )

        # Stream chunks
        try:
            for chunk in completion_iterable:
                try:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None) if delta else None
                    
                    if not content:
                        try:
                            content = chunk.choices[0].message.content
                        except Exception:
                            content = None
                    
                    if content:
                        yield content
                        if run_manager:
                            run_manager.on_llm_new_token(content)
                except Exception:
                    s = str(chunk)
                    if s:
                        yield s
                        if run_manager:
                            run_manager.on_llm_new_token(s)
        except Exception as e:
            # Fallback to non-streaming
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=float(self.temperature),
                    max_tokens=int(self.max_tokens),
                    top_p=float(self.top_p),
                    stream=False,
                    stop=stop or []
                )
                
                if hasattr(resp, 'choices') and len(resp.choices) > 0:
                    content = resp.choices[0].message.content
                    if content:
                        yield content
                        if run_manager:
                            run_manager.on_llm_new_token(content)
            except Exception as fallback_error:
                raise RuntimeError(f"Groq streaming failed: {e}. Fallback also failed: {fallback_error}")
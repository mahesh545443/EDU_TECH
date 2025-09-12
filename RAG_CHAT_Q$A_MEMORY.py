# RAG_CHAT_Q$A_MEMORY.py — final robust version for Streamlit Cloud
import os

# disable aggressive file watching (must be set before importing streamlit)
os.environ["WATCHFILES_DISABLE"] = "1"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

# --- Ensure a newer SQLite is used (pysqlite3-binary must be in requirements.txt) ---
import sys
try:
    import pysqlite3.dbapi2 as _pysqlite_dbapi
    sys.modules["sqlite3"] = _pysqlite_dbapi
except Exception:
    try:
        import pysqlite3 as _pysqlite3
        sys.modules["sqlite3"] = _pysqlite3
    except Exception:
        # fallback: system sqlite3 (may be old)
        pass

# suppress noisy torch internal warning (optional)
import warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# standard imports
import gc
import uuid
import json
import requests
import streamlit as st
from pathlib import Path
from typing import Union

# LangChain community imports (after sqlite fix)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Embeddings: prefer langchain-huggingface; fallback to others
try:
    from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
    EmbeddingsClass = _HuggingFaceEmbeddings
except Exception:
    try:
        from langchain.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings2
        EmbeddingsClass = _HuggingFaceEmbeddings2
    except Exception:
        # final fallback
        from langchain_community.embeddings import SentenceTransformerEmbeddings as _SentenceTransformerEmbeddings
        EmbeddingsClass = _SentenceTransformerEmbeddings

# (optional) import sentence-transformers for fallback direct encoding
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMER_AVAILABLE = True
except Exception:
    _SENTENCE_TRANSFORMER_AVAILABLE = False


class PDFProcessor:
    def __init__(self, vector_db_root: Union[str, Path] = None):
        # Initialize embeddings object (handle different constructor signatures)
        try:
            self.embeddings = EmbeddingsClass(model_name="all-MiniLM-L6-v2")
        except TypeError:
            self.embeddings = EmbeddingsClass("all-MiniLM-L6-v2")

        # If sentence-transformers is available, prepare a direct model as a robust fallback
        if _SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._st_model = None
        else:
            self._st_model = None

        # Where vector DBs are stored (relative to app directory)
        if vector_db_root is None:
            self.vector_db_root = Path(os.getcwd()) / "analytics_chroma"
        else:
            self.vector_db_root = Path(vector_db_root)
        self.vector_db_root.mkdir(parents=True, exist_ok=True)

        # OpenRouter API key (from Streamlit secrets or env)
        api_key = None
        try:
            api_key = st.secrets.get("OPENROUTER_API_KEY") if getattr(st, "secrets", None) else None
        except Exception:
            api_key = None
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_api_key = api_key

        # OpenRouter endpoint
        self.openrouter_endpoint = "https://openrouter.ai/api/v1/chat/completions"

    # ---------------- Embedding helper ----------------
    def _embedding_fn(self, texts):
        """
        Robust embedding function that accepts either a single string or a list of strings.
        Returns list-of-embeddings (each embedding is a list of floats) OR single embedding if input was string.
        """
        single_input = False
        if isinstance(texts, str):
            single_input = True
            texts = [texts]

        # Try common LangChain embedding method names in priority
        try:
            if hasattr(self.embeddings, "embed_documents"):
                embs = self.embeddings.embed_documents(texts)
            elif hasattr(self.embeddings, "embed_queries") and hasattr(self.embeddings, "embed_query"):
                # some implementations have embed_query for single queries
                embs = [self.embeddings.embed_query(t) for t in texts]
            elif hasattr(self.embeddings, "embed_query"):
                embs = [self.embeddings.embed_query(t) for t in texts]
            elif hasattr(self.embeddings, "embed"):
                # generic embed method
                embs = self.embeddings.embed(texts)
            else:
                raise AttributeError("No known embed method on embeddings object")
        except Exception:
            # Fallback: use sentence-transformers directly if available
            if self._st_model is not None:
                try:
                    arr = self._st_model.encode(texts)
                    # arr may be ndarray; convert to list of lists
                    if hasattr(arr, "tolist"):
                        embs = arr.tolist()
                    else:
                        embs = [list(x) for x in arr]
                except Exception as e:
                    raise RuntimeError(f"Fallback sentence-transformers encoding failed: {e}")
            else:
                raise RuntimeError("Embedding failed and no sentence-transformers fallback available.")

        # Ensure consistent return: list of lists or single list if input was single string
        if single_input:
            return embs[0] if isinstance(embs, list) else embs
        return embs

    # ---------------- Chroma helpers ----------------
    def _create_chroma_from_documents(self, documents, persist_directory: str):
        """
        Try multiple ways to create a Chroma vectorstore from documents to support multiple versions.
        """
        errors = []
        # Primary: prefer embedding_function param
        try:
            return Chroma.from_documents(documents=documents, embedding_function=self._embedding_fn, persist_directory=persist_directory)
        except Exception as e:
            errors.append(("embedding_function_from_documents", repr(e)))

        # Fallback: older wrapper that accepts 'embedding' (embeddings object)
        try:
            return Chroma.from_documents(documents=documents, embedding=self.embeddings, persist_directory=persist_directory)
        except Exception as e:
            errors.append(("embedding_from_documents", repr(e)))

        # Last fallback: try direct constructor + add_documents if available
        try:
            chroma_inst = Chroma(persist_directory=persist_directory)
            if hasattr(chroma_inst, "add_documents"):
                chroma_inst.add_documents(documents)
                chroma_inst.persist()
                return chroma_inst
        except Exception as e:
            errors.append(("constructor_add_documents", repr(e)))

        raise RuntimeError(f"Could not create Chroma DB. Attempts: {errors}")

    def _load_chroma_from_persist(self, persist_directory: str):
        """
        Try multiple ways to load an existing Chroma DB from disk.
        """
        errors = []
        # Preferred factory
        try:
            return Chroma.from_persist_directory(persist_directory=persist_directory, embedding_function=self._embedding_fn)
        except Exception as e:
            errors.append(("from_persist_embedding_function", repr(e)))

        # Older factory with embedding object
        try:
            return Chroma.from_persist_directory(persist_directory=persist_directory, embedding=self.embeddings)
        except Exception as e:
            errors.append(("from_persist_embedding", repr(e)))

        # Fallback to constructor forms
        try:
            return Chroma(persist_directory=persist_directory, embedding_function=self._embedding_fn)
        except Exception as e:
            errors.append(("constructor_embedding_function", repr(e)))
        try:
            return Chroma(persist_directory=persist_directory, embedding=self.embeddings)
        except Exception as e:
            errors.append(("constructor_embedding", repr(e)))

        raise RuntimeError(f"Could not load Chroma DB. Attempts: {errors}")

    # ---------------- PDF processing ----------------
    def _save_uploaded_pdf(self, uploaded_file) -> str:
        suffix = ".pdf"
        tmp_name = f"{uuid.uuid4().hex}{suffix}"
        tmp_path = self.vector_db_root / "tmp_uploads"
        tmp_path.mkdir(parents=True, exist_ok=True)
        file_path = tmp_path / tmp_name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(file_path)

    def process_pdf(self, pdf_source: Union[str, "UploadedFile"]):
        """Process PDF and create or reuse a Chroma vector DB. Returns vector_path or None."""
        is_temp = False
        pdf_path = None
        try:
            if hasattr(pdf_source, "read"):
                pdf_path = self._save_uploaded_pdf(pdf_source)
                is_temp = True
            else:
                pdf_path = str(pdf_source)

            if not os.path.exists(pdf_path):
                st.error(f"PDF not found at: {pdf_path}")
                return None

            db_name = Path(pdf_path).stem
            vector_path = str(self.vector_db_root / db_name)
            flag_path = os.path.join(vector_path, "processed.flag")

            if os.path.exists(flag_path):
                st.info("This PDF was already processed. Reusing existing vector DB.")
                return vector_path

            # Load + split
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            # Ensure vector folder exists before writing
            os.makedirs(vector_path, exist_ok=True)

            # Build Chroma DB using embeddings object or embedding_function (robust)
            vectordb = self._create_chroma_from_documents(chunks, vector_path)
            # persist (if supported)
            try:
                vectordb.persist()
            except Exception:
                pass

            # mark processed
            with open(flag_path, "w") as f:
                f.write("processed")

            return vector_path

        except Exception as e:
            st.error(f"[PROCESS ERROR] {e}")
            return None

        finally:
            if is_temp and pdf_path:
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass
            gc.collect()

    # ---------------- OpenRouter call ----------------
    def _call_openrouter(self, messages: list, model: str = "openai/gpt-4o-mini", max_tokens: int = 600):
        """
        Direct HTTP call to OpenRouter chat completions. Returns (ok: bool, content_or_error).
        """
        if not self.openrouter_api_key:
            return False, "OPENROUTER_API_KEY not configured (Streamlit Secrets or env)."

        payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
        headers = {"Authorization": f"Bearer {self.openrouter_api_key}", "Content-Type": "application/json"}

        try:
            resp = requests.post(self.openrouter_endpoint, headers=headers, json=payload, timeout=60)
        except Exception as e:
            return False, f"Network error when calling OpenRouter: {e}"

        if resp.status_code != 200:
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            return False, f"OpenRouter returned {resp.status_code}: {body}"

        try:
            data = resp.json()
        except Exception as e:
            return False, f"Invalid JSON response from OpenRouter: {e}"

        # extract content from known shapes
        try:
            choice = data["choices"][0]
            msg = choice.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return True, content
                if isinstance(content, dict) and "text" in content:
                    return True, content["text"]
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        if isinstance(part, str):
                            parts.append(part)
                        elif isinstance(part, dict):
                            text = part.get("text") or part.get("content")
                            if isinstance(text, str):
                                parts.append(text)
                    return True, "\n".join(parts)
            if "text" in choice:
                return True, choice["text"]
            return False, f"Unexpected response shape: {json.dumps(choice)[:400]}"
        except Exception as e:
            return False, f"Could not extract content: {e}"

    # ---------------- Querying ----------------
    def query_document(self, vector_path: str, question: str, chat_history: list, model_name: str = "openai/gpt-4o-mini"):
        if not os.path.exists(vector_path):
            return "Vector DB not found at specified path."

        # Load existing Chroma DB safely
        try:
            vectordb = self._load_chroma_from_persist(vector_path)
        except Exception as e:
            return f"[VECTOR ERROR] Could not load Chroma DB: {e}"

        # Perform retrieval robustly (support similarity_search or retriever)
        try:
            if hasattr(vectordb, "similarity_search"):
                results = vectordb.similarity_search(question, k=3)
            else:
                if hasattr(vectordb, "as_retriever"):
                    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                    if hasattr(retriever, "get_relevant_documents"):
                        results = retriever.get_relevant_documents(question)
                    else:
                        # some retrievers are callable
                        results = retriever(question)
                else:
                    return "[VECTOR ERROR] Vector store does not support retrieval."
        except Exception as e:
            return f"[VECTOR QUERY ERROR] {e}"

        if not results:
            return "No relevant information found."

        context = "\n\n".join([r.page_content for r in results])

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a subject expert assistant. Answer using the context from the PDF like a textbook solution. "
                    "Include LaTeX formulas ($$...$$) where relevant; be precise and step-by-step."
                )
            }
        ]

        for q, a in chat_history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

        ok, content = self._call_openrouter(messages=messages, model=model_name, max_tokens=600)
        if not ok:
            return f"[MODEL ERROR] {content}"
        return content


# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="📄 PDF Q&A Bot", layout="wide")
st.title("📄 PDF Q&A Assistant")

processor = PDFProcessor()

# Sidebar / settings
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose model", ["openai/gpt-4o-mini", "openai/gpt-4o", "mistralai/mixtral-8x7b-instruct"])
st.sidebar.markdown("Vector DBs saved under `analytics_chroma/` in the app folder.")
st.sidebar.markdown("Add `OPENROUTER_API_KEY` in Streamlit Secrets before using the model.")

# Upload / sample selection
st.subheader("Upload or select a PDF")
uploaded = st.file_uploader("Upload a PDF (or use sample below)", type=["pdf"])

sample_path = Path("sample_pdfs")
sample_pdfs = []
if sample_path.exists():
    for p in sample_path.glob("*.pdf"):
        sample_pdfs.append(str(p))
if sample_pdfs:
    chosen = st.selectbox("Or choose a sample PDF from repo", ["-- select --"] + sample_pdfs)
else:
    chosen = "-- none --"

# Use session_state to keep selected vector DB across reruns
if "vector_path" not in st.session_state:
    st.session_state["vector_path"] = None

if uploaded is not None:
    if st.button("Process uploaded PDF"):
        with st.spinner("Processing uploaded PDF..."):
            vector_path = processor.process_pdf(uploaded)
            if vector_path:
                st.success("PDF processed and vector DB created.")
                st.session_state["vector_path"] = vector_path
else:
    if chosen != "-- select --" and chosen != "-- none --":
        if st.button("Process selected repo PDF"):
            with st.spinner("Processing PDF from repo..."):
                vector_path = processor.process_pdf(chosen)
                if vector_path:
                    st.success("PDF processed and vector DB created.")
                    st.session_state["vector_path"] = vector_path

# show existing processed DBs (only directories with processed.flag)
existing_dbs = sorted(
    [str(p) for p in Path(processor.vector_db_root).iterdir() if p.is_dir() and (p / "processed.flag").exists()]
)
if existing_dbs:
    db_choice = st.selectbox("Or pick an existing processed PDF (vector DB)", ["-- pick --"] + existing_dbs)
    if db_choice != "-- pick --":
        st.session_state["vector_path"] = db_choice

st.subheader("💬 Ask questions (once a vector DB is selected)")
vector_path = st.session_state.get("vector_path")
if vector_path:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question about the PDF:", key="qa_input")
    if st.button("Ask"):
        if user_input.strip():
            with st.spinner("Thinking..."):
                answer = processor.query_document(
                    vector_path,
                    user_input,
                    st.session_state.get("chat_history", []),
                    model_name=model_choice
                )
                st.session_state.chat_history.append((user_input, answer))
        else:
            st.warning("Please type a question.")

    if st.session_state.get("chat_history"):
        st.markdown("### Conversation")
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")
else:
    st.info("No vector DB selected / processed yet. Upload or choose a PDF and click 'Process'.")

# housekeeping
if st.button("Clear chat history"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")






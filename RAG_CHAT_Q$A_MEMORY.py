# RAG_CHAT_Q$A_MEMORY.py — final robust version for Streamlit Cloud
# RAG_CHAT_Q$A_MEMORY.py — final robust version (embedding adapter + chroma compatibility)
import os
import sys
import gc
import uuid
import json
import requests
import streamlit as st
from pathlib import Path
from typing import Union, List, Any
import warnings

# -------------------------
# Environment / SQLite fix
# -------------------------
os.environ["WATCHFILES_DISABLE"] = "1"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

# Prefer pysqlite3 if available (provides newer sqlite version required by chroma)
try:
    import pysqlite3.dbapi2 as _pysqlite_dbapi
    sys.modules["sqlite3"] = _pysqlite_dbapi
except Exception:
    try:
        import pysqlite3 as _pysqlite3
        sys.modules["sqlite3"] = _pysqlite3
    except Exception:
        pass  # fallback to system sqlite3 (may be old)

# suppress noisy torch message
warnings.filterwarnings("ignore", message=".*torch.classes.*")

# -------------------------
# Imports for RAG pipeline
# -------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Try preferred embedding packages in an order that covers common installs
EmbeddingsClass = None
try:
    # recommended: langchain-huggingface
    from langchain_huggingface import HuggingFaceEmbeddings as _HuggingFaceEmbeddings
    EmbeddingsClass = _HuggingFaceEmbeddings
except Exception:
    try:
        # fallback: langchain's old location
        from langchain.embeddings import HuggingFaceEmbeddings as _HuggingFaceEmbeddings2
        EmbeddingsClass = _HuggingFaceEmbeddings2
    except Exception:
        try:
            # fallback: community sentence-transformer wrapper
            from langchain_community.embeddings import SentenceTransformerEmbeddings as _SentenceTransformerEmbeddings
            EmbeddingsClass = _SentenceTransformerEmbeddings
        except Exception:
            EmbeddingsClass = None

# Optional direct sentence-transformers model fallback
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMER_AVAILABLE = True
except Exception:
    _SENTENCE_TRANSFORMER_AVAILABLE = False


# -------------------------
# EmbeddingAdapter: provides embed_documents, embed_query, and is callable
# -------------------------
class EmbeddingAdapter:
    def __init__(self, embeddings_obj: Any = None, st_model: Any = None):
        """
        embeddings_obj: an embeddings object (e.g., HuggingFaceEmbeddings, SentenceTransformerEmbeddings)
        st_model: direct SentenceTransformer model (optional fallback)
        """
        self.emb = embeddings_obj
        self.st_model = st_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Return list of embedding vectors for list of texts.
        """
        if self.emb is not None:
            # common langchain methods
            if hasattr(self.emb, "embed_documents"):
                return self.emb.embed_documents(texts)
            if hasattr(self.emb, "embed"):
                # some wrappers implement embed(list[str]) -> list[vector]
                try:
                    return self.emb.embed(texts)
                except Exception:
                    pass
            if hasattr(self.emb, "embed_query"):
                # embed_query usually for single, but we can map it
                return [self.emb.embed_query(t) for t in texts]
            if hasattr(self.emb, "embed_queries"):
                return self.emb.embed_queries(texts)

        # fallback to sentence-transformers
        if self.st_model is not None:
            arr = self.st_model.encode(texts)
            if hasattr(arr, "tolist"):
                return arr.tolist()
            return [list(x) for x in arr]

        raise RuntimeError("No embedding method available (embed_documents/embed/embed_query).")

    def embed_query(self, text: str) -> List[float]:
        """
        Return embedding for a single query string.
        """
        if self.emb is not None:
            if hasattr(self.emb, "embed_query"):
                return self.emb.embed_query(text)
            if hasattr(self.emb, "embed"):
                out = self.emb.embed([text])
                if isinstance(out, list):
                    return out[0]
                return out
            if hasattr(self.emb, "embed_documents"):
                out = self.emb.embed_documents([text])
                return out[0] if isinstance(out, list) else out

        if self.st_model is not None:
            arr = self.st_model.encode([text])
            if hasattr(arr, "tolist"):
                return arr[0].tolist()
            return list(arr[0])

        raise RuntimeError("No embed_query or fallback available.")

    def __call__(self, texts: Union[str, List[str]]):
        """
        Make adapter callable: accepts a single string or list and returns embeddings.
        This satisfies embedding_function=callable expectations.
        """
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True
        out = self.embed_documents(texts)
        return out[0] if single else out


# -------------------------
# PDF Processor
# -------------------------
class PDFProcessor:
    def __init__(self, vector_db_root: Union[str, Path] = None):
        # init embeddings object (try various constructors)
        st_model = None
        if _SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                st_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                st_model = None

        embeddings_obj = None
        if EmbeddingsClass is not None:
            try:
                embeddings_obj = EmbeddingsClass(model_name="all-MiniLM-L6-v2")
            except TypeError:
                try:
                    embeddings_obj = EmbeddingsClass("all-MiniLM-L6-v2")
                except Exception:
                    embeddings_obj = None

        # create adapter (object and callable)
        self.embedding_adapter = EmbeddingAdapter(embeddings_obj=embeddings_obj, st_model=st_model)
        self.embedding_fn = self.embedding_adapter  # callable
        self.embedding_obj = self.embedding_adapter  # object with embed_documents/embed_query

        # vector DB root
        self.vector_db_root = Path(vector_db_root or (Path.cwd() / "analytics_chroma"))
        self.vector_db_root.mkdir(parents=True, exist_ok=True)

        # OpenRouter key (streamlit secrets or env)
        api_key = None
        try:
            api_key = st.secrets.get("OPENROUTER_API_KEY") if getattr(st, "secrets", None) else None
        except Exception:
            api_key = None
        if not api_key:
            api_key = os.environ.get("OPENROUTER_API_KEY")
        self.openrouter_api_key = api_key

        self.openrouter_endpoint = "https://openrouter.ai/api/v1/chat/completions"

    # ---- file helpers ----
    def _save_uploaded_pdf(self, uploaded_file) -> str:
        suffix = ".pdf"
        tmp_name = f"{uuid.uuid4().hex}{suffix}"
        tmp_path = self.vector_db_root / "tmp_uploads"
        tmp_path.mkdir(parents=True, exist_ok=True)
        file_path = tmp_path / tmp_name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(file_path)

    # ---- create vector DB: try embedding_function first, then embedding object ----
    def _create_chroma_from_documents(self, documents, persist_directory: str):
        errors = []
        # try embedding_function (callable)
        try:
            return Chroma.from_documents(documents=documents, embedding_function=self.embedding_fn, persist_directory=persist_directory)
        except Exception as e:
            errors.append(("embedding_function", repr(e)))

        # try embedding object
        try:
            return Chroma.from_documents(documents=documents, embedding=self.embedding_obj, persist_directory=persist_directory)
        except Exception as e:
            errors.append(("embedding_object", repr(e)))

        # try constructor + add_documents
        try:
            c = Chroma(persist_directory=persist_directory)
            if hasattr(c, "add_documents"):
                c.add_documents(documents)
                try:
                    c.persist()
                except Exception:
                    pass
                return c
        except Exception as e:
            errors.append(("constructor_add_documents", repr(e)))

        raise RuntimeError(f"Could not create Chroma DB. Attempts: {errors}")

    # ---- load persisted chroma safely ----
    def _load_chroma_from_persist(self, persist_directory: str):
        errors = []
        # try from_persist_directory with embedding_function
        try:
            return Chroma.from_persist_directory(persist_directory=persist_directory, embedding_function=self.embedding_fn)
        except Exception as e:
            errors.append(("from_persist_embedding_function", repr(e)))

        # try from_persist_directory with embedding object
        try:
            return Chroma.from_persist_directory(persist_directory=persist_directory, embedding=self.embedding_obj)
        except Exception as e:
            errors.append(("from_persist_embedding", repr(e)))

        # try constructor with embedding_function
        try:
            return Chroma(persist_directory=persist_directory, embedding_function=self.embedding_fn)
        except Exception as e:
            errors.append(("constructor_embedding_function", repr(e)))

        # try constructor with embedding object
        try:
            return Chroma(persist_directory=persist_directory, embedding=self.embedding_obj)
        except Exception as e:
            errors.append(("constructor_embedding", repr(e)))

        raise RuntimeError(f"Could not load Chroma DB. Attempts: {errors}")

    # ---- process PDF ----
    def process_pdf(self, pdf_source: Union[str, "UploadedFile"]):
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

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            os.makedirs(vector_path, exist_ok=True)

            vectordb = self._create_chroma_from_documents(chunks, vector_path)
            try:
                vectordb.persist()
            except Exception:
                pass

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

    # ---- OpenRouter call ----
    def _call_openrouter(self, messages: list, model: str = "openai/gpt-4o-mini", max_tokens: int = 600):
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
                            txt = part.get("text") or part.get("content")
                            if isinstance(txt, str):
                                parts.append(txt)
                    return True, "\n".join(parts)
            if "text" in choice:
                return True, choice["text"]
            return False, f"Unexpected response shape: {json.dumps(choice)[:400]}"
        except Exception as e:
            return False, f"Could not extract content: {e}"

    # ---- query existing vector DB ----
    def query_document(self, vector_path: str, question: str, chat_history: list, model_name: str = "openai/gpt-4o-mini"):
        if not os.path.exists(vector_path):
            return "Vector DB not found at specified path."

        try:
            vectordb = self._load_chroma_from_persist(vector_path)
        except Exception as e:
            return f"[VECTOR ERROR] Could not load Chroma DB: {e}"

        try:
            if hasattr(vectordb, "similarity_search"):
                results = vectordb.similarity_search(question, k=3)
            else:
                if hasattr(vectordb, "as_retriever"):
                    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
                    if hasattr(retriever, "get_relevant_documents"):
                        results = retriever.get_relevant_documents(question)
                    else:
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








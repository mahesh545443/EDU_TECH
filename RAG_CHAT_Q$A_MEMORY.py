# RAG_CHAT_Q$A_MEMORY.py
import os
# disable aggressive file watching (must be set before importing streamlit)
os.environ["WATCHFILES_DISABLE"] = "1"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import gc
import uuid
import json
import requests
import streamlit as st
from pathlib import Path
from typing import Union

# LangChain community imports (for loaders & vector DB)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


class PDFProcessor:
    def __init__(self, vector_db_root: Union[str, Path] = None):
        # embedding model (sentence-transformers)
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

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
        self.openrouter_api_key = api_key  # may be None (we handle that later)

        # OpenRouter endpoint (Chat Completions)
        self.openrouter_endpoint = "https://openrouter.ai/api/v1/chat/completions"

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

            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding_function=self.embeddings,
                persist_directory=vector_path
            )
            vectordb.persist()

            os.makedirs(vector_path, exist_ok=True)
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

    def _call_openrouter(self, messages: list, model: str = "openai/gpt-4o-mini", max_tokens: int = 600):
        """
        Direct HTTP call to OpenRouter chat completions.
        Returns (success_bool, content_or_error_message).
        """
        if not self.openrouter_api_key:
            return False, "OPENROUTER_API_KEY not configured (in Streamlit secrets or env)."

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }

        try:
            resp = requests.post(self.openrouter_endpoint, headers=headers, json=payload, timeout=60)
        except Exception as e:
            return False, f"Network error when calling OpenRouter: {e}"

        if resp.status_code != 200:
            # return short debug info (avoid leaking sensitive info)
            try:
                body = resp.json()
            except Exception:
                body = resp.text
            return False, f"OpenRouter returned {resp.status_code}: {body}"

        try:
            data = resp.json()
        except Exception as e:
            return False, f"Invalid JSON response from OpenRouter: {e}"

        # Extract content: supports OpenAI-compatible response shape
        try:
            choice = data["choices"][0]
            # Newer OpenRouter responses use choice['message']['content']
            if "message" in choice and "content" in choice["message"]:
                return True, choice["message"]["content"]
            # Fallback to text field
            if "text" in choice:
                return True, choice["text"]
            return False, f"Unexpected response shape: {json.dumps(choice)[:400]}"
        except Exception as e:
            return False, f"Could not extract content: {e}"

    def query_document(self, vector_path: str, question: str, chat_history: list, model_name: str = "openai/gpt-4o-mini"):
        if not os.path.exists(vector_path):
            return "Vector DB not found at specified path."

        vectordb = Chroma(
            persist_directory=vector_path,
            embedding_function=self.embeddings
        )

        results = vectordb.similarity_search(question, k=3)
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

st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose model",
    ["openai/gpt-4o-mini", "openai/gpt-4o", "mistralai/mixtral-8x7b-instruct"]
)
st.sidebar.markdown("Vector DB storage: `analytics_chroma/` in app folder.")
st.sidebar.markdown("Add `OPENROUTER_API_KEY` in Streamlit Secrets before asking questions.")

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

vector_path = None
if uploaded is not None:
    if st.button("Process uploaded PDF"):
        with st.spinner("Processing uploaded PDF..."):
            vector_path = processor.process_pdf(uploaded)
            if vector_path:
                st.success("PDF processed and vector DB created.")
else:
    if chosen != "-- select --" and chosen != "-- none --":
        if st.button("Process selected repo PDF"):
            with st.spinner("Processing PDF from repo..."):
                vector_path = processor.process_pdf(chosen)
                if vector_path:
                    st.success("PDF processed and vector DB created.")

existing_dbs = sorted([str(p) for p in Path(processor.vector_db_root).iterdir() if p.is_dir()])
if existing_dbs:
    db_choice = st.selectbox("Or pick an existing processed PDF (vector DB)", ["-- pick --"] + existing_dbs)
    if db_choice != "-- pick --":
        vector_path = db_choice

st.subheader("💬 Ask questions (once a vector DB is selected)")
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

if st.button("Clear chat history"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")






# RAG_CHAT_Q$A_MEMORY.py
import os
# Disable aggressive file watching (fixes inotify errors in hosted envs)
os.environ["WATCHFILES_DISABLE"] = "1"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import gc
import tempfile
import uuid
import streamlit as st
from openai import OpenAI
from pathlib import Path
from typing import Union

# LangChain community imports (correct for recent langchain versions)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


# ===============================
# PDF Processor Class
# ===============================
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

        # OpenRouter/OpenAI client
        api_key = st.secrets.get("OPENROUTER_API_KEY") if hasattr(st, "secrets") else os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            st.warning("OPENROUTER_API_KEY not found in Streamlit secrets or env. Add it before calling the model.")
            self.client = None
        else:
            # initialize client and set OpenRouter base URL
            self.client = OpenAI(api_key=api_key)
            # openai.OpenAI client supports setting base_url attribute
            self.client.base_url = "https://openrouter.ai/api/v1"

    def _save_uploaded_pdf(self, uploaded_file) -> str:
        """Save a streamlit uploaded file to a temp file and return file path."""
        suffix = ".pdf"
        tmp_name = f"{uuid.uuid4().hex}{suffix}"
        tmp_path = self.vector_db_root / "tmp_uploads"
        tmp_path.mkdir(parents=True, exist_ok=True)
        file_path = tmp_path / tmp_name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(file_path)

    def process_pdf(self, pdf_source: Union[str, "UploadedFile"]):
        """
        Accept either:
         - a filesystem path (str)
         - a Streamlit UploadedFile object
        Returns path to persisted vector DB (folder) or None on error.
        """
        # handle uploaded file
        is_temp = False
        try:
            if hasattr(pdf_source, "read"):  # uploaded file
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

            # Load and split
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            # Build Chroma DB (persisted)
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding_function=self.embeddings,
                persist_directory=vector_path
            )
            vectordb.persist()

            # mark processed
            os.makedirs(vector_path, exist_ok=True)
            with open(flag_path, "w") as f:
                f.write("processed")

            return vector_path

        except Exception as e:
            st.error(f"[PROCESS ERROR] {e}")
            return None

        finally:
            # cleanup and GC
            if is_temp:
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass
            gc.collect()

    def query_document(self, vector_path: str, question: str, chat_history: list, model_name: str = "openai/gpt-4o-mini"):
        """Retrieve context from Chroma and ask the model via OpenRouter/OpenAI client."""
        if not os.path.exists(vector_path):
            return "Vector DB not found at specified path."

        if self.client is None:
            return "OpenRouter API client not configured. Add OPENROUTER_API_KEY to Streamlit secrets."

        vectordb = Chroma(
            persist_directory=vector_path,
            embedding_function=self.embeddings
        )

        results = vectordb.similarity_search(question, k=3)
        if not results:
            return "No relevant information found."

        context = "\n\n".join([r.page_content for r in results])

        # Build system + history + user prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a subject expert assistant. Answer using the context from the PDF like a textbook solution. "
                    "1) Include formulas in LaTeX ($$...$$) when relevant. "
                    "2) Use step-by-step bullets when explaining. "
                    "3) Be precise and academic."
                )
            }
        ]

        for q, a in chat_history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

        try:
            resp = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=600
            )
            # safe extraction (depends on client version)
            content = None
            try:
                content = resp.choices[0].message.content
            except Exception:
                # fallback: sometimes response text is in resp.choices[0].text
                content = getattr(resp.choices[0], "text", str(resp))
            return content
        except Exception as e:
            return f"[MODEL ERROR] {e}"


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="📄 PDF Q&A Bot", layout="wide")
st.title("📄 PDF Q&A Assistant")

processor = PDFProcessor()

# Sidebar: small controls
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose model",
    ["openai/gpt-4o-mini", "openai/gpt-4o", "mistralai/mixtral-8x7b-instruct"]
)

st.sidebar.markdown("**Vector DB storage:** saved to `analytics_chroma/` inside the app folder.")
st.sidebar.markdown("Add `OPENROUTER_API_KEY` in Streamlit Secrets before asking questions.")

# Upload or use repo file
st.subheader("Upload or select a PDF")
uploaded = st.file_uploader("Upload a PDF (or use sample below)", type=["pdf"])

# If you want to point to a PDF in repo, change this path or add file to repo.
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

# If user previously processed, show available vector DBs
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
                # save history
                st.session_state.chat_history.append((user_input, answer))
        else:
            st.warning("Please type a question.")

    # show history
    if st.session_state.get("chat_history"):
        st.markdown("### Conversation")
        for q, a in st.session_state.chat_history:
            st.markdown(f"**You:** {q}")
            st.markdown(f"**Bot:** {a}")
            st.markdown("---")
else:
    st.info("No vector DB selected / processed yet. Upload or choose a PDF and click 'Process'.")

# housekeeping: free memory button
if st.button("Clear chat history"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")



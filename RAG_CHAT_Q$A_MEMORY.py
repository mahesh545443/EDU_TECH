import os
import gc
import streamlit as st
from openai import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings


# ===============================
# PDF Processor Class
# ===============================
class PDFProcessor:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db_path = r"C:\Users\User\analytics_chroma" #create one vector folder 
        os.makedirs(self.vector_db_path, exist_ok=True)

        # ✅ Initialize OpenRouter client directly
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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"  # 🔑 replace with your valid key
        )

    def process_pdf(self, pdf_path):
        """Process PDF and create/update vector DB"""
        db_name = os.path.splitext(os.path.basename(pdf_path))[0]
        vector_path = os.path.join(self.vector_db_path, db_name)

        flag_path = os.path.join(vector_path, "processed.flag")
        if os.path.exists(flag_path):
            return vector_path

        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=vector_path
            )
            vectordb.persist()

            with open(flag_path, "w") as f:
                f.write("processed")

            return vector_path

        except Exception as e:
            st.error(f"[ERROR] {str(e)}")
            return None
        finally:
            if 'documents' in locals(): del documents
            if 'chunks' in locals(): del chunks
            if 'vectordb' in locals(): del vectordb
            gc.collect()

    def query_document(self, vector_path, question, chat_history, model_name="openai/gpt-4o-mini"):
        """Retrieve context & generate answer using OpenRouter"""
        if not os.path.exists(vector_path):
            raise ValueError("Vector DB not found at specified path")

        vectordb = Chroma(
            persist_directory=vector_path,
            embedding_function=self.embeddings
        )

        # Retrieve top 3 relevant chunks
        results = vectordb.similarity_search(question, k=3)
        if not results:
            return "No relevant information found."

        context = " ".join([r.page_content for r in results])

        # Build conversation with strong instructions
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a subject expert assistant. "
                    "Always answer using the context from the PDF like a **textbook solution**. "
                    "Follow these rules:\n"
                    "1. Always include **formulas** in LaTeX format (surrounded by $$ ... $$) if relevant.\n"
                    "2. Give step-by-step explanation in clear bullet points if needed.\n"
                    "3. Keep answers precise, like in engineering/physics books.\n"
                    "4. If user asks follow-up (like 'what is delta'), explain in **clear, academic style**.\n"
                    "5. Never skip formulas if they are in the PDF context."
                )
            }
        ]

        for q, a in chat_history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        })

        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=500
        )

        return response.choices[0].message.content


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="📄 PDF Q&A Bot", layout="wide")
st.title("📄 PDF Q&A Assistant")

processor = PDFProcessor()

# ✅ Fixed PDF path as u have
pdf_path = r"C:\Users\User\Downloads\keph201.pdf"
vector_path = processor.process_pdf(pdf_path)

# Sidebar settings
st.sidebar.header("⚙️ Settings")
model_choice = st.sidebar.selectbox("Choose Model", ["openai/gpt-4o-mini", "openai/gpt-4o", "mistralai/mixtral-8x7b-instruct"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.subheader("💬 Chat with your PDF")

user_input = st.text_input("Ask a question about the PDF:", "")
if st.button("Ask") and user_input.strip():
    with st.spinner("Thinking..."):
        answer = processor.query_document(
            vector_path,
            user_input,
            st.session_state.chat_history,
            model_name=model_choice
        )
        st.session_state.chat_history.append((user_input, answer))

# Display chat history
for q, a in st.session_state.chat_history:
    st.markdown(f"** You:** {q}")
    st.markdown(f"** Bot:** {a}")
    st.markdown("---")

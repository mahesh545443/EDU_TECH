import os
import warnings
import pandas as pd
import streamlit as st
from PIL import Image
from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document

# === CONFIG ===
EXCEL_FOLDER = r"C:\Users\User\analytics_excel"
CHROMA_DB_PATH = r"C:\Users\User\analytics_vector_db_new"
ROW_LIMIT = 10

os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

# === STREAMLIT SETUP ===
st.set_page_config(page_title="AARD AI Assistant", page_icon="", layout="wide")

# === CUSTOM CSS ===
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #fffef5;
    }

    .main {
        padding: 2rem;
        background-color: #fffef5;
    }

    h1 {
        color: #e6b800;
        font-weight: 700;
    }

    .stTextInput>div>div>input {
        border: 2px solid #e6b800;
        padding: 10px;
        font-size: 16px;
    }

    .stButton>button {
        background-color: #e6b800;
        color: black;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        border: none;
        border-radius: 6px;
        font-size: 16px;
    }

    .block-container {
        padding-top: 20px;
    }

    .custom-box {
        background-color: #fff9d6;
        padding: 20px;
        border-left: 6px solid #e6b800;
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .response-box {
        background-color: #fcf6d9;
        padding: 20px;
        border-radius: 6px;
        font-size: 17px;
        line-height: 1.6;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# === LOGO ===
logo_path = r"C:\Users\User\OneDrive\Documents\Downloads\analytics_avenue_for_research_and_development_logo.jpeg"
try:
    logo = Image.open(logo_path)
    st.image(logo, width=150)
except Exception as e:
    st.warning(f"⚠️ Unable to load logo: {e}")

# === HEADER ===
st.markdown("<h1>AARD AI Assistant</h1>", unsafe_allow_html=True)
st.write("A smart assistant to query insights from your Excel/CSV data. Built for modern business intelligence.")

# === WARNINGS ===
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# === EMBEDDING + CLIENT SETUP ===
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
client = OpenAI()

# === VECTOR DB BUILDER ===
@st.cache_resource
def build_vector_db():
    all_documents = []

    for root, dirs, files in os.walk(EXCEL_FOLDER):
        for file in files:
            if file.endswith(".csv") or file.endswith(".xlsx"):
                try:
                    path = os.path.join(root, file)
                    df = pd.read_csv(path, nrows=ROW_LIMIT) if file.endswith(".csv") else pd.read_excel(path, nrows=ROW_LIMIT)
                    df = df.astype(str)

                    # Clean phone-like columns
                    for col in df.columns:
                        if "phone" in col.lower():
                            df[col] = df[col].apply(lambda x: str(x).split('.')[0] if '.' in str(x) else str(x))

                    for i, row in df.iterrows():
                        row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                        all_documents.append(Document(page_content=row_text, metadata={"source": file, "row": i}))

                except Exception as e:
                    st.warning(f"Failed to load {file}: {e}")

    texts = text_splitter.split_documents(all_documents)
    db = Chroma.from_documents(
        documents=texts,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH
    )
    db.persist()
    return db

# === LOAD VECTOR DB ===
if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
    with st.spinner("Creating vector database from Excel files..."):
        db = build_vector_db()
else:
    db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

retriever = db.as_retriever(search_kwargs={"k": 5})

# === PROMPT TEMPLATE ===
prompt_template = """
You are a professional data analytics assistant.
Your task is to answer questions strictly based on the structured data extracted from Excel files.
Use only the context provided to form clear and concise responses.
If the context does not contain enough information, reply that the answer is not available in the data.

Question: {question}

Context:
{context}

Answer:
"""

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

# === QUESTION INPUT UI ===
st.markdown("<div class='custom-box'><h4>Ask a question based on your data</h4></div>", unsafe_allow_html=True)

user_input = st.text_input("Enter your query")
ask_btn = st.button("Submit")

if ask_btn and user_input:
    with st.spinner("Searching and analyzing data..."):
        try:
            docs = retriever.invoke(user_input)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = custom_prompt.format(question=user_input, context=context)

            response = client.chat.completions.create(
                model="google/gemma-3-27b-it:free",
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response.choices[0].message.content.strip()
            st.markdown(f"<div class='response-box'>{answer}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error during response generation: {e}")

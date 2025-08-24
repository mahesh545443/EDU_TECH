import os
import io
import base64
import fitz  # PyMuPDF
import torch
import numpy as np
import warnings
from PIL import Image
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity

# === Load environment variables ===
os.environ["OPENAI_API_KEY"] = "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"

# === Suppress warnings ===
warnings.filterwarnings("ignore")

# === Initialize CLIP Model ===
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# === Embedding functions ===
def embed_image(image_data):
    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGB")
    else:
        image = image_data

    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

def embed_text(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze().numpy()

# === PDF to Embeddings ===
pdf_path = r"C:\\Users\\User\\Downloads\\multimodal_sample (1).pdf"
doc = fitz.open(pdf_path)

all_docs = []
all_embeddings = []
image_data_store = {}
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

for i, page in enumerate(doc):
    text = page.get_text()
    if text.strip():
        temp_doc = Document(page_content=text, metadata={"page": i, "type": "text"})
        text_chunks = splitter.split_documents([temp_doc])
        for chunk in text_chunks:
            embedding = embed_text(chunk.page_content)
            all_embeddings.append(embedding)
            all_docs.append(chunk)

    for img_index, img in enumerate(page.get_images(full=True)):
        try:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_id = f"page_{i}_img_{img_index}"

            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            image_data_store[image_id] = img_base64

            embedding = embed_image(pil_image)
            all_embeddings.append(embedding)

            image_doc = Document(
                page_content=f"[Image: {image_id}]",
                metadata={"page": i, "type": "image", "image_id": image_id}
            )
            all_docs.append(image_doc)

        except Exception as e:
            print(f"Error processing image {img_index} on page {i}: {e}")

doc.close()

# === FAISS Vector Store ===
embeddings_array = np.array(all_embeddings)
vector_store = FAISS.from_embeddings(
    text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
    embedding=None,
    metadatas=[doc.metadata for doc in all_docs]
)

# === OpenAI GPT-4V Setup ===
from openai import OpenAI

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENAI_API_KEY"])

# === Retrieval ===
def retrieve_multimodal(query, k=5):
    query_embedding = embed_text(query)
    results = vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)
    return results

# === Message Builder ===
def create_multimodal_message(query, retrieved_docs):
    content = []
    content.append({"type": "text", "text": f"Question: {query}\n\nContext:"})

    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]

    if text_docs:
        text_context = "\n\n".join([
            f"[Page {doc.metadata['page']}]: {doc.page_content}" for doc in text_docs
        ])
        content.append({"type": "text", "text": f"Text excerpts:\n{text_context}\n"})

    for doc in image_docs:
        image_id = doc.metadata.get("image_id")
        if image_id in image_data_store:
            content.append({"type": "text", "text": f"\n[Image from page {doc.metadata['page']}]:\n"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data_store[image_id]}"}
            })

    content.append({"type": "text", "text": "\n\nPlease answer the question based on the provided text and images."})
    return {"role": "user", "content": content}

# === Pipeline ===
def multimodal_pdf_rag_pipeline(query):
    context_docs = retrieve_multimodal(query, k=5)
    message = create_multimodal_message(query, context_docs)
    response = client.chat.completions.create(model="openai/gpt-4.1", messages=[message],max_tokens=1024)

    print(f"\nRetrieved {len(context_docs)} documents:")
    for doc in context_docs:
        doc_type = doc.metadata.get("type", "unknown")
        page = doc.metadata.get("page", "?")
        if doc_type == "text":
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  - Text from page {page}: {preview}")
        else:
            print(f"  - Image from page {page}")
    print("\n")

    return response.choices[0].message.content

# === Example Execution ===
# === your multimodal rag pipeline with streamlit is now fully ready ===

# 💡 suggestions to improve your implementation:
# 1. fix indentation: move Streamlit app under `if __name__ == "__main__":`
# 2. dynamically upload PDF via Streamlit (instead of hardcoded path)
# 3. cache vectorstore to avoid recomputation
# 4. use session state to avoid reprocessing

# 🔧 fixed version below (simplified):

if __name__ == "__main__":
    import streamlit as st
    from PIL import Image
    import base64
    import io  # Required for decoding image bytes
    import os

    # === PAGE SETUP ===
    st.set_page_config(page_title="📊 Analytics Avenue - MultiModal RAG", layout="wide")

    # === LOGO LOAD ===
    logo_path = r"C:\Users\User\OneDrive\Documents\Downloads\analytics_avenue_for_research_and_development_logo.jpeg"
    try:
        logo = Image.open(logo_path)
        st.image(logo, width=120)
    except FileNotFoundError:
        st.warning("⚠️ Logo image not found. Please check the path.")

    # === HEADER ===
    st.markdown("""
        <h1 style='color:#d4a300;'>Analytics Avenue - Multimodal PDF Q&A</h1>
        <p style='font-size:18px;'>Ask questions from text & images inside your PDF!</p>
        <hr style="border: 1px solid #d4a300;">
    """, unsafe_allow_html=True)

    # === USER INPUT ===
    user_query = st.text_input("🔍 Enter your question:", placeholder="e.g. What does the chart on page 1 show?")

    if st.button("🧠 Ask"):
        if user_query.strip():
            with st.spinner("🔎 Analyzing document..."):
                results = retrieve_multimodal(user_query, k=5)
                message = create_multimodal_message(user_query, results)
                response = client.chat.completions.create(
                    model="openai/gpt-4.1",
                    messages=[message],
                    max_tokens=1024
                )
                answer = response.choices[0].message.content

            # === SHOW RETRIEVED CONTEXT ===
            st.markdown("### 📘 Retrieved Context:")
            for doc in results:
                doc_type = doc.metadata.get("type", "unknown")
                page = doc.metadata.get("page", "?")
                if doc_type == "text":
                    st.markdown(f"**📄 Page {page} Text:**")
                    st.write(doc.page_content)
                elif doc_type == "image":
                    image_id = doc.metadata.get("image_id")
                    if image_id in image_data_store:
                        try:
                            image_data = image_data_store[image_id]
                            image_bytes = base64.b64decode(image_data)
                            image = Image.open(io.BytesIO(image_bytes))
                            st.markdown(f"**🖼️ Page {page} Image:**")
                            st.image(image, use_column_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ Could not load image from page {page}: {e}")
                    else:
                        st.warning(f"⚠️ Image ID {image_id} not found in image_data_store.")

            # === DISPLAY FINAL ANSWER ===
            st.markdown("### 🧠 Answer:")
            st.markdown(f"<div style='background-color:#fff5cc; padding:15px; border-radius:10px; font-size:16px;'>{answer}</div>", unsafe_allow_html=True)

        else:
            st.warning("⚠️ Please enter a question to continue.")


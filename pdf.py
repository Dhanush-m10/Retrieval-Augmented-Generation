import os
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from groq import Groq
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# ==============================
# 1️⃣ Load API Key from Streamlit Secrets
# ==============================

try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("❌ GROQ_API_KEY not found in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ==============================
# 2️⃣ Streamlit UI
# ==============================

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📘")

st.title("📘 Intelligent PDF RAG Assistant")
st.write("Upload a PDF (scanned or digital) and ask questions.")

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==============================
# 3️⃣ Upload PDF
# ==============================

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# ==============================
# 4️⃣ Text Extraction
# ==============================

def extract_text_from_pdf(pdf_file):

    text = ""
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for page in doc:

        page_text = page.get_text()

        # If page has no text → use OCR
        if not page_text.strip():
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            try:
                page_text = pytesseract.image_to_string(img)
            except:
                page_text = ""

        text += page_text + "\n"

    return text


# ==============================
# 5️⃣ Create Vectorstore
# ==============================

def create_vectorstore(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    documents = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


# ==============================
# 6️⃣ Process PDF
# ==============================

if uploaded_file:

    if "vectorstore" not in st.session_state:

        with st.spinner("Processing PDF..."):

            text = extract_text_from_pdf(uploaded_file)
            st.session_state.vectorstore = create_vectorstore(text)

        st.success("✅ PDF processed successfully!")

    query = st.chat_input("Ask a question about your PDF")

    if query:

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(query)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the context provided.
If the answer is not found, say:
"I don't know based on the document."

Context:
{context}

Question:
{query}

Answer:
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        answer = response.choices[0].message.content

        # Save chat history
        st.session_state.chat_history.append(("user", query))
        st.session_state.chat_history.append(("assistant", answer))

    # Display chat
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

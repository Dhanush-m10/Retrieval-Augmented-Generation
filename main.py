import os
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from groq import Groq

# ==============================
# 1️⃣ Load Environment Variables
# ==============================

# Try loading Streamlit secrets first
if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    # fallback for local development
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please add it to Streamlit secrets or .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# ==============================
# 2️⃣ Streamlit UI
# ==============================

st.set_page_config(page_title="Importance of ML RAG App", page_icon="🌍")
st.title("🌍 RAG App - Importance of Machine Learning")
st.write("Ask questions about the **Importance of Machine Learning**.")


# ==============================
# 3️⃣ Load and Process Documents
# ==============================

@st.cache_resource
def load_vectorstore():

    with open("documents.txt", "r", encoding="utf-8") as f:
        text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(text)

    documents = [Document(page_content=chunk) for chunk in chunks]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore


vectorstore = load_vectorstore()


# ==============================
# 4️⃣ Query Input
# ==============================

query = st.text_input("Enter your question:")

if query:

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a data scientist answering questions about the importance of machine learning.

Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    st.subheader("📌 Answer")
    st.write(answer)

    # Optional: show sources
    # st.subheader("📚 Sources Used")
    # for doc in docs:
    #     st.write("-", doc.page_content)

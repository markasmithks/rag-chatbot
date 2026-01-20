import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
VECTORSTORE_PATH = "vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- App title ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 RAG Chatbot (LangChain Docs)")

st.write(
    "Ask questions about LangChain and LangGraph documentation. "
    "Answers are grounded in the retrieved documents."
)

# --- Load embeddings and vector store ---
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )
    db = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return db

db = load_vectorstore()

# --- User input ---
query = st.text_input("Enter your question:")

# --- Retrieval ---
if query:
    with st.spinner("Searching documents..."):
        results = db.similarity_search(query, k=3)

    st.subheader("Retrieved Context")
    for i, doc in enumerate(results, 1):
        st.markdown(f"**Source {i}: {doc.metadata.get('source')}**")
        st.write(doc.page_content[:500])
        st.markdown("---")

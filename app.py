import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from llm import load_llm, generate_answer, needs_retrieval  


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

def load_model():
    return load_llm()

llm = load_model()

db = load_vectorstore()

# --- User input ---
query = st.text_input("Enter your question:")

# --- Retrieval ---
if query:
    with st.spinner("Deciding how to answer..."):
        retrieve = needs_retrieval(query, llm)
    # Conservative routing policy: always retrieve for domain-specific queries
    technical_keywords = ["langchain", "langgraph", "memory", "agent", "persistence", "implement"]
    if any(k in query.lower() for k in technical_keywords):
        retrieve = True


    if retrieve:
        with st.spinner("Searching documents..."):
            results = db.similarity_search(query, k=3)

        context = "\n\n".join(
            f"Source {i}:\n{doc.page_content}"
            for i, doc in enumerate(results, 1)
        )

        with st.spinner("Generating grounded answer..."):
            answer = generate_answer(query, context, llm)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, doc in enumerate(results, 1):
            st.markdown(f"**Source {i}: {doc.metadata.get('source')}**")
            st.write(doc.page_content[:300])
            st.markdown("---")

    else:
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, "", llm)

        st.subheader("Answer")
        st.write(answer)

        st.caption("Answered without document retrieval.")

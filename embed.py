from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document

from ingest import load_markdown_documents, split_documents

VECTORSTORE_PATH = Path("vectorstore")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    print("Loading documents...")
    documents = load_markdown_documents()
    chunks = split_documents(documents)
    print(f"Loaded {len(chunks)} chunks")

    print("Loading embedding model...")
    #model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = HuggingFaceEmbeddings(
         model_name=EMBEDDING_MODEL_NAME
         )
    print("Creating FAISS vector store...")
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    VECTORSTORE_PATH.mkdir(exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"Vector store saved to: {VECTORSTORE_PATH}")


if __name__ == "__main__":
    main()

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


DOCS_PATH = Path("data/docs")

def load_markdown_documents():
    documents = []

    for md_file in DOCS_PATH.rglob("*.md"):
        loader = TextLoader(str(md_file), encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = md_file.name
            doc.metadata["path"] = str(md_file)

        documents.extend(docs)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            " "
        ]
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_markdown_documents()
    chunks = split_documents(docs)

    print(f"Loaded {len(docs)} markdown documents")
    print(f"Created {len(chunks)} chunks")

    # Optional: inspect one chunk
    sample = chunks[0]
    print("\nSample chunk metadata:")
    print(sample.metadata)
    print("\nSample content (truncated):")
    print(sample.page_content[:500])

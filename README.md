# RAG Chatbot

A retrieval-augmented chatbot that answers user questions by grounding responses in external
technical documentation. This project is designed as a portfolio-grade example of
document ingestion, vector-based retrieval, and LLM-assisted answer generation.

The system follows a modular RAG architecture, with clear separation between ingestion,
embedding, retrieval, generation, and user interface layers.

---

## Project Goals

- Demonstrate a clean Retrieval-Augmented Generation (RAG) pipeline
- Ground answers in external documents using vector similarity search
- Use local, open-source embeddings and models for reproducibility and portability
- Provide a minimal, transparent UI that exposes retrieved sources
- Explore practical RAG refinement techniques (prompting and context construction)
- Serve as a foundation for more advanced agentic behavior

---

## Current Features

- Markdown document ingestion and chunking
- Hugging Face MiniLM embeddings (`all-MiniLM-L6-v2`)
- FAISS-based vector similarity search
- Verified semantic retrieval over a curated document corpus
- LLM-based answer generation using a local Hugging Face model (`flan-t5-base`)
- Prompt and context refinement to improve grounding and relevance
- Streamlit UI for interactive querying with transparent source attribution

---

## Tech Stack

- Python 3.11
- LangChain (community + Hugging Face integrations)
- Sentence-Transformers (MiniLM)
- Hugging Face Transformers (FLAN-T5)
- FAISS (vector search)
- Streamlit (UI)
- Conda + pip for environment management

---

## Project Structure

```text
rag-chatbot/
├── ingest.py          # Load and chunk source documents
├── embed.py           # Create FAISS vector store from chunks
├── llm.py             # LLM loading and grounded answer generation
├── app.py             # Streamlit UI for retrieval + generation
├── data/
│   └── docs/          # Curated Markdown document corpus
├── vectorstore/       # Generated FAISS index (gitignored)
├── environment.yml
├── requirements.txt
└── README.md
```
---

## Corpus

This demo uses a curated subset of publicly available LangChain and LangGraph documentation,
stored as Markdown files. Documents are chunked, embedded, and indexed for retrieval.

Current corpus includes:
- LangGraph overview and core concepts
- LangGraph persistence and workflow documentation
- LangChain memory and model documentation

---

## Setup

```bash
conda env create -f environment.yml
conda activate rag-chatbot

python ingest.py
python embed.py
streamlit run app.py
```

---

## Development Workflow

Feature development was performed on isolated Git branches and merged into `main` once
functionality was complete and validated.

For example, LLM-based answer generation and RAG refinement were developed on a dedicated
feature branch before being merged back into `main`, keeping the primary branch stable
throughout development.

---

## Notes on Version Control

Generated artifacts such as embeddings, FAISS indexes, and local secrets are intentionally
excluded from version control. All source code and configuration required to reproduce the
system is tracked in the repository.
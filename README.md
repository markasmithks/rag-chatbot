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

## How It Works

This project implements a standard Retrieval-Augmented Generation (RAG) pipeline with
explicit separation between ingestion, retrieval, and answer generation.

### 1. Document Ingestion
Source documents (Markdown files) are loaded and split into semantically coherent chunks
using a recursive text splitter. Each chunk retains metadata indicating its source file.

### 2. Embedding and Indexing
All document chunks are embedded using a local Hugging Face sentence-transformer
(`all-MiniLM-L6-v2`). The resulting vectors are stored in a FAISS index to enable fast
semantic similarity search.

### 3. Retrieval
When a user submits a query, the application performs a similarity search against the
FAISS index to retrieve the most relevant document chunks. These chunks form the
retrieval context for answer generation.

### 4. Agentic Routing (Retrieval Decision)
Before performing retrieval, the system evaluates whether a user query
requires consulting external documentation.

This decision is made using a lightweight LLM-based classifier that
assesses whether the question depends on project-specific technical
details (e.g., implementation or configuration behavior).

To reduce the risk of under-retrieval, routing follows a conservative
policy that biases toward retrieval for domain-specific technical
queries. This ensures that documentation-backed answers are preferred
whenever relevant, while still allowing direct responses for general
or non-corpus questions.

### 5. Answer Generation
The retrieved context is passed to a local instruction-tuned language model
(`google/flan-t5-base`) along with the user’s question. The model is explicitly instructed
to answer the question using only the provided context and to avoid unrelated information.

Prompt structure and context formatting were refined to improve grounding and reduce
off-topic responses when multiple concepts appear in the retrieved documents.

### 6. User Interface
A Streamlit application (`app.py`) serves as the user interface. It displays the generated
answer along with the underlying source excerpts, providing transparency into how each
response was produced.


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

---
## Optional OpenAI Configuration

By default, the application runs fully locally using Hugging Face models.
A lightweight local model is always used for agentic routing decisions,
and local generation (`google/flan-t5-base`) is used for answering questions
unless explicitly configured otherwise.

OpenAI-based answer generation is supported as an **optional backend** and
must be explicitly enabled by the user.

### Enabling OpenAI (Optional)

To enable OpenAI for answer generation, create a local `.env` file with:

```env
OPENAI_API_KEY=your_api_key_here
LLM_BACKEND=openai

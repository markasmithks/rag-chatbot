## Notes
This project excludes generated embeddings, environment files, and secrets from version control.

# RAG Chatbot

A retrieval-augmented conversational chatbot that answers user questions using external documents.
The project is designed as a portfolio-grade example of document grounding, retrieval pipelines,
and lightweight agentic decision-making.

## Project Goals
- Demonstrate a clean Retrieval-Augmented Generation (RAG) architecture
- Leverage external documents as a knowledge source
- Implement a simple agentic routing step to decide when retrieval is required
- Provide a usable, minimal UI for interaction

## Tech Stack
- Python 3.11
- LangChain
- Streamlit
- OpenAI-compatible LLMs
- PDF-based document ingestion

## Current Status
This project is under active development.
Initial focus is on document ingestion and retrieval before UI and deployment.

## Setup

```bash
conda env create -f environment.yml
conda activate rag-chatbot

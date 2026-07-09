# Research Paper RAG Chatbot

A multimodal Retrieval-Augmented Generation (RAG) chatbot that enables users to query research papers in natural language. It ingests papers from arXiv or local PDFs, extracts text, figures, and tables, stores embeddings in ChromaDB, and generates source-grounded responses using an LLM.

## Features

- Semantic search over research papers
- Paper ingestion from arXiv IDs, search queries, or local PDFs
- Automatic extraction of text, figures, and tables
- Figure and table-aware question answering
- Multi-paper retrieval and comparison
- Persistent ChromaDB vector storage
- FastAPI backend for chatbot APIs

## Tech Stack

- Python
- FastAPI
- LangChain
- ChromaDB
- Hugging Face Embeddings
- Groq LLM
- PyMuPDF
- Transformers
- Moondream Vision Model

## Workflow

```text
Research Paper
      │
      ▼
Text / Figures / Tables Extraction
      │
      ▼
Embedding Generation
      │
      ▼
ChromaDB Vector Store
      │
      ▼
Semantic Retrieval
      │
      ▼
Groq LLM
      │
      ▼
Grounded Answer with Citations
```

## Example Queries

- Summarize this paper.
- Explain Figure 3 in X paper.
- What are the key contributions?
- Compare two research papers.
- What datasets were used?
- Explain Table 2 in X paper,

# AI Powered Equity Research Tool
## Project Overview

This project is a web-based tool designed for the efficient analysis of news articles through a natural language interface. It implements a Retrieval-Augmented Generation (RAG) pipeline to provide users with synthesized answers to complex queries based on the content of user-provided URLs.

The application utilizes a hybrid AI architecture, leveraging a local, open-source model for cost-effective text embedding and a powerful large language model (LLM) for high-fidelity answer generation.

## Key Features

* **Multi-Source Analysis:** Ingests and processes content from multiple news article URLs simultaneously.
* **Natural Language Q&A:** Allows users to ask questions in plain English and receive context-aware, source-backed answers.
* **Hybrid AI Architecture:**
    * **Local Embeddings:** Employs a Hugging Face `sentence-transformers` model for efficient, no-cost text embedding.
    * **Cloud LLM:** Utilizes the Google Gemini API for state-of-the-art generative capabilities.
* **Graceful Fallback:** In the absence of a Gemini API key, the system defaults to a robust similarity search, returning the most relevant text excerpts.
* **Data Validation:** Includes a pre-processing step to validate URL accessibility and format, ensuring system stability.

## Technical Architecture

The application is built on a Retrieval-Augmented Generation (RAG) pipeline orchestrated by the LangChain framework.

1.  **Data Ingestion:** Content is scraped from user-provided URLs using LangChain's `WebBaseLoader`.
2.  **Text Processing:** The extracted documents are segmented into smaller, uniform chunks using the `RecursiveCharacterTextSplitter`.
3.  **Embedding & Indexing:** A local Hugging Face model generates vector embeddings for each text chunk. These embeddings are indexed and stored in a `FAISS` (Facebook AI Similarity Search) in-memory vector store.
4.  **Retrieval:** When a user submits a query, it is embedded using the same model. A similarity search is performed on the `FAISS` index to retrieve the most semantically relevant document chunks.
5.  **Generation:** The original user query and the retrieved context chunks are passed to the Google Gemini model, which synthesizes a final, coherent answer.

## Technology Stack

* **Application Framework:** Streamlit
* **AI Orchestration:** LangChain
* **Embedding Model:** Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
* **Generative LLM:** Google Gemini (`gemini-1.5-flash-latest`)
* **Vector Database:** FAISS

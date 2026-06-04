# Document Question Answering System using LangChain RAG

A Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and ask natural language questions about their contents. The system extracts text, generates vector embeddings using Google's Gemini Embedding Model, stores them in a FAISS vector database, and retrieves relevant context to generate accurate responses.

## Features

* Multi-PDF upload and processing
* Semantic search using FAISS
* Google Gemini-powered question answering
* Context-aware responses using RAG
* Interactive Streamlit interface

## Tech Stack

* Python
* Streamlit
* LangChain
* Google Gemini API
* FAISS
* PyPDF2

## How It Works

1. Upload one or more PDF documents.
2. Text is extracted and split into chunks.
3. Chunks are converted into embeddings.
4. Embeddings are stored in a FAISS vector database.
5. Relevant chunks are retrieved for each user query.
6. Gemini generates answers using the retrieved context.

## Key Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Semantic Search
* LLM Integration
* Document Intelligence

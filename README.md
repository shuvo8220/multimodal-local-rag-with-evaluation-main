# multimodal-local-rag-with-evaluation-main
A multimodal RAG system that intelligently processes PDFs containing text, images, and tables. It uses advanced chunking, semantic and hybrid search, local LLM generation, and vision-based analysis to deliver accurate, context-aware answers through a clean, interactive web interface.
Multimodal RAG System – Intelligent PDF Understanding Platform

# Key Features
# Multimodal Document Processing
Semantic Text Chunking: Structured text segmentation with title-aware organization
Image Understanding: Visual interpretation using GPT-4o-mini
Table Extraction: Intelligent parsing, restructuring, and summarization using DeepSeek
Layout-Aware Processing: High-resolution page structure and element detection

# Advanced Retrieval Capabilities

Semantic Search: Embedding-based similarity search
Keyword Search (BM25): Precise lexical term matching
Hybrid Search: Optimized combination (70% semantic + 30% keyword) for maximum accuracy

# AI-Driven Generation

Local LLM via Ollama: Uses deepseek-r1:1.5b for contextual answer generation
Citation-Aware Responses: Generates structured outputs with referenced document chunks
Multimodal Reasoning: Integrates insights from text, images, and tables

# User-Friendly Interface
Drag-and-Drop PDF Upload
Real-Time Document Processing Status
Interactive Querying and Instant Responses
Clean, Professional Gradio UI


# Project Architecture

├── app.py                 # Main Gradio web application
├── get_chunks.py          # PDF text, image, and table extraction
├── ingestion_pipeline.py  # Vector database creation (FAISS)
├── retrieval_pipeline.py  # Semantic, keyword, and hybrid search logic
├── generator_pipeline.py  # LLM-based answer generation
├── helper.py              # Utilities and embedding helpers
├── requirements.txt
├── README.md
├── .env
└── faiss_index/           # Vector store (auto-generated)

# Prerequisites
# System Requirements
Python 3.8+
Minimum 8 GB RAM
At least 5 GB free disk space

# External Dependencies
Ollama (Local LLM + Embeddings)
OpenAI API Key (for image understanding via GPT-4o-mini)
Tesseract OCR (for extracting text from images inside PDFs)

# Installation Guide
1. Clone the Repository
   git clone <your-repository-url>
   cd multimodal-rag-system

2.Create and Activate a Virtual Environment
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
3.Install Dependencies
   pip install -r requirements.txt
4.Install Ollama
5.Pull Required Models
   ollama pull deepseek-r1:1.5b
   ollama pull nomic-embed-text
   
6.Install Tesseract OCR

7.Add Environment Variables
   OPENAI_API_KEY=your_openai_api_key
   OLLAMA_BASE_URL=http://localhost:11434

# How to Run
1.Start Ollama
   ollama serve
2.Launch the Application
   python app.py

# Example Queries
“What is the main topic of the document?”
“Summarize the tables presented in the PDF.”
“What information do the images convey?”
“Explain the methodology described in the paper.”
“List the datasets used in this research.”


# Acknowledgments
Unstructured.io
Ollama
LangChain
Gradio
OpenAI
FAISS







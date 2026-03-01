# 🤖 Premium Multi-Source Agentic RAG

A state-of-the-art, high-performance RAG system built with **FastAPI**, **Next.js**, **Qdrant**, and **Google Gemini 2.5 Flash**. This system is designed for enterprise-grade document intelligence, featuring multi-source orchestration and hybrid retrieval.

## 🧠 Architectural Design (Agentic Flow)

The system follows a modular architecture inspired by advanced agentic patterns. Unlike traditional RAG, this pipeline dynamically decides which "Knowledge Collections" to consult based on the user's intent.

### 1. Ingestion Engine (The Knowledge Base)
The system supports four distinct knowledge streams, each with optimized ingestion parameters:
- **Research Papers**: Deep PDF processing using `PyPDFLoader` and `RecursiveCharacterTextSplitter` with 1000-character chunks.
- **Knowledge Base**: Business/Internal documentation processed from Markdown files.
- **Code Docs**: Technical documentation and repositories, optimized for code-snippet retention.
- **FAQ Data**: Structured Question-Answer pairs for high-precision retrieval.

**Vector Engine**: Powered by **Qdrant Cloud**, using the `all-MiniLM-L6-v2` embedding model (384-dimensional vectors) with Cosine similarity.

### 2. Intelligent Agentic Planner
Instead of searching a single flat database, the **Planner** acts as a traffic controller.
- **Instant Keyword Routing**: Using a high-speed keyword dictionary derived from the notebook patterns, the planner identifies which of the 4 collections are most relevant to the query.
- **Parallel Execution**: Once collections are selected, the system triggers simultaneous searches across all target databases, minimizing wait times.

### 3. High-Performance Retrieval Pipeline
- **Unified Embedding**: To save API costs and reduce latency, the system generates a single query embedding and reuses it across all selected collection searches.
- **Hybrid Retrieval**: Combines Dense Vector search (Semantic) with **BM25 Okapi** reranking (Keyword-based). This ensures that specific technical terms (like "Layer Normalization") are prioritized even if semantic similarity is broad.
- **Expanded Context**: The system retrieves the top 15 candidates and reranks them to provide the LLM with the 10 most relevant context blocks.

### 4. Expert Synthesis (Gemini 2.5 Flash)
The final stage uses **Gemini 2.5 Flash** with a meticulously tuned system prompt to produce **Strictly Plain Prose**.
- **Natural Conversation**: Information is synthesized into 2–3 coherent paragraphs.
- **No Formatting Clutter**: Explicitly forbids markdown headers, bullet points, and source citations, creating a conversational and human-like response.
- **Zero-CORS Latency**: The architecture is designed for real-time streaming to the Next.js frontend on port 3000.

### 5. Professional UI (Frontend)
The Next.js frontend features a modern, enterprise-grade interface.
- **Clean Light Theme**: A professional off-white/teal color palette designed for readability and focus.
- **Enterprise Aesthetics**: Minimalist design with subtle shadows, removing gamer-style glow effects and dark backgrounds.
- **Interactive Experience**: Streaming chat responses and drag-and-drop PDF uploads.

## 🚥 Quick Start

### Backend (Port 8000)
1. `cd backend`
2. `pip install -r requirements.txt`
3. Configure `.env` with `GEMINI_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY`.
4. Run: `python -m uvicorn main:app --host 0.0.0.0 --port 8000`

### Frontend (Port 3000)
1. `cd frontend`
2. `npm install`
3. Run: `npm run dev -- -p 3000`

## 📁 Source Code Map

- `backend/main.py`: FastAPI application & Core Agent logic.
- `backend/retrieval/retriever.py`: Hybrid search & BM25 implementation.
- `backend/ingestion/ingestion.py`: Pipeline for multi-source PDF/Text indexing.
- `backend/retrieval/planner.py`: The Keyword-based Agentic Router.
- `frontend/app/page.tsx`: Premium Chat UI with streaming support.

## 📝 License
MIT License

# 🤖 Multi-Source Agentic RAG System

A production-ready RAG system built with **FastAPI**, **Next.js**, **Qdrant Cloud**, and **Google Gemini**. Features multi-source orchestration, hybrid retrieval, and a professional enterprise UI.

## 🔗 Live Demo

- **Frontend**: Deployed on [Vercel](https://vercel.com)
- **Backend**: [https://mutlisouce-agentic-rag-system-1.onrender.com](https://mutlisouce-agentic-rag-system-1.onrender.com)

## 🏗️ Architecture

```
User Question → Agentic Planner → Hybrid Retrieval → Gemini 2.5 Flash → Streamed Prose Answer
                     ↓                    ↓
            Collection Routing    Vector Search + BM25 Reranking
            (4 Knowledge Pools)   (Qdrant Cloud + Google Embeddings)
```

### 1. Ingestion Engine
- **PDF Processing**: `PyPDFLoader` + `RecursiveCharacterTextSplitter` (1000-char chunks, 200 overlap)
- **4 Collections**: Research Papers, Knowledge Base, Code Docs, FAQ Data
- **Vector Engine**: **Qdrant Cloud** with **Google Gemini Embeddings** (`gemini-embedding-001`, 3072-dimensional vectors, Cosine similarity)

### 2. Intelligent Planner
- **Keyword Routing**: Instantly routes queries to the most relevant collections
- **Parallel Execution**: Searches multiple collections simultaneously

### 3. Hybrid Retrieval Pipeline
- **Unified Embedding**: Single query embedding reused across all collection searches
- **Hybrid Search**: Dense Vector (semantic) + **BM25 Okapi** reranking (keyword-based)
- **Deep Context**: Top 15 candidates → BM25 reranked → Top 10 sent to LLM

### 4. Expert Synthesis (Gemini 2.5 Flash)
- **Plain Prose Output**: Strict system prompt produces natural 2–3 paragraph answers
- **No Formatting Clutter**: Forbids markdown, bullet points, and source citations
- **Real-time Streaming**: Responses stream directly to the frontend

### 5. Professional UI
- **Clean Light Theme**: Off-white/teal palette, no dark backgrounds or glow effects
- **Enterprise Design**: Minimal shadows, Inter font, responsive layout
- **Interactive**: Drag-and-drop PDF upload, streaming chat, sidebar with collection stats

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Google Gemini 2.5 Flash |
| **Embeddings** | Google Gemini Embedding API (`gemini-embedding-001`) |
| **Vector DB** | Qdrant Cloud |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Next.js 16, React 19, TailwindCSS v4 |
| **Deployment** | Render (backend) + Vercel (frontend) |

## 🚀 Quick Start

### Backend (Port 8000)
```bash
cd backend
pip install -r requirements.txt
# Configure .env with GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend (Port 3000)
```bash
cd frontend
npm install
npm run dev
```

## ☁️ Deployment

### Backend → Render
| Field | Value |
|-------|-------|
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port 10000` |
| Python Version | 3.11 (via `runtime.txt`) |

**Env Variables**: `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`

### Frontend → Vercel
| Field | Value |
|-------|-------|
| Root Directory | `frontend` |
| Framework | Next.js (auto-detected) |
| Env Variable | `NEXT_PUBLIC_API_URL` = your Render backend URL |

## 📁 Project Structure

```
backend/
├── main.py                 # FastAPI app, chat & upload endpoints
├── core/
│   ├── llm.py              # Gemini 2.5 Flash configuration
│   ├── embeddings.py       # Google Embedding API (gemini-embedding-001)
│   ├── qdrant_client.py    # Qdrant Cloud connection
│   └── config.py           # Environment variable loader
├── retrieval/
│   ├── retriever.py        # Hybrid vector + BM25 search
│   └── planner.py          # Keyword-based collection router
├── ingestion/
│   └── ingestion.py        # PDF processing & Qdrant ingestion
├── memory/
│   └── memory.py           # In-memory chat history
├── cache/
│   └── cache.py            # Response caching
├── requirements.txt        # Python dependencies (no PyTorch)
└── runtime.txt             # Python 3.11 for Render

frontend/
├── app/
│   ├── page.tsx            # Chat UI with streaming
│   ├── layout.tsx          # Root layout + Inter font
│   └── globals.css         # Light theme design system
└── package.json            # Next.js 16 + React 19
```

## 📝 License
MIT License

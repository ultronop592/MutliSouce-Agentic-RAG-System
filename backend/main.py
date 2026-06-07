"""
Multi-Source Agentic RAG — FastAPI Application
===============================================
Entry point for the production RAG backend.

Agent pipeline (coordinated by the Orchestrator):
  Router Agent  →  Memory Agent  →  Retrieval Agent
                                         ↓
                                   Reranker Agent
                                         ↓
                                   Guard (confidence)
                                         ↓
                                   Answer Agent (streamed)

All heavy logic lives in agents/ — main.py only handles HTTP routing,
startup/shutdown, and error boundaries.
"""

import os
import shutil
import tempfile
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from agents.orchestrator import orchestrator
from retrieval.bm25_index import bm25_manager
from memory.memory import chat_memory
from cache.cache import response_cache
from ingestion.ingestion import ingest_pdf, ensure_collections, COLLECTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify Qdrant, pre-load BM25 indexes from existing documents."""
    logger.info("=== RAG Backend starting up ===")

    # 1. Ensure Qdrant collections exist
    try:
        ensure_collections()
        logger.info("Qdrant collections verified")
    except Exception as e:
        logger.warning("Qdrant connection warning: %s", e)

    # 2. Pre-load BM25 indexes from ALL existing Qdrant documents.
    #    Without this, BM25 search returns nothing until the first upload.
    try:
        counts = bm25_manager.refresh_all(COLLECTIONS)
        for col, n in counts.items():
            if n > 0:
                logger.info("BM25 index loaded: '%s' — %d docs", col, n)
        logger.info("BM25 indexes ready")
    except Exception as e:
        logger.warning("BM25 index warning: %s", e)

    logger.info("=== RAG Backend ready ===")
    yield
    logger.info("=== RAG Backend shutting down ===")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Source Agentic RAG API",
    description=(
        "Production-ready multi-agent RAG with hybrid search (semantic + BM25 + RRF), "
        "Gemini reranking, hallucination guard, semantic cache, and conversation memory."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"
    collections: list[str] | None = None



class UploadResponse(BaseModel):
    filename: str
    chunks_ingested: int
    collection: str
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "Multi-Source Agentic RAG",
        "version": "3.0.0",
        "status": "running",
        "pipeline": [
            "RouterAgent",
            "MemoryAgent",
            "RetrievalAgent (semantic + BM25 + RRF)",
            "RerankerAgent (Gemini pointwise)",
            "HallucinationGuard",
            "AnswerAgent (streamed)",
        ],
        "endpoints": ["/chat", "/upload", "/collections", "/memory", "/health"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "3.0.0"}


@app.get("/collections")
async def list_collections():
    """List all Qdrant collections with point counts and BM25 index status."""
    from core.qdrant_client import qdrant
    try:
        ensure_collections()
        result = qdrant.get_collections()
        collection_list = []
        for col in result.collections:
            try:
                col_info = qdrant.get_collection(col.name)
                collection_list.append({
                    "name": col.name,
                    "points_count": col_info.points_count,
                    "vectors_count": col_info.vectors_count,
                    "bm25_docs": bm25_manager.doc_count(col.name),
                    "bm25_ready": bm25_manager.has_index(col.name),
                })
            except Exception:
                collection_list.append({
                    "name": col.name,
                    "points_count": 0,
                    "vectors_count": 0,
                    "bm25_docs": 0,
                    "bm25_ready": False,
                })
        return {"collections": collection_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@app.get("/memory/{session_id}")
async def get_memory(session_id: str):
    """Return conversation history for a session."""
    history_str = chat_memory.format_history(session_id)
    return {
        "session_id": session_id,
        "turn_count": chat_memory.turn_count(session_id),
        "history": history_str,
    }


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    collection: str = "research_papers",
):
    """Upload a PDF and ingest it into Qdrant. Automatically refreshes BM25 index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        chunks_count = ingest_pdf(temp_path, collection)

        return UploadResponse(
            filename=file.filename,
            chunks_ingested=chunks_count,
            collection=collection,
            message=(
                f"Successfully ingested {chunks_count} chunks from '{file.filename}'. "
                f"BM25 index refreshed."
            ),
        )
    except Exception as e:
        logger.error("Ingestion failed for %s: %s", file.filename, e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint — full multi-agent RAG pipeline.

    Pipeline:
      1. Router Agent   — classify query intent
      2. Cache check    — semantic similarity lookup (fast path)
      3. Memory Agent   — semantic memory lookup (fast path)
      4. Retrieval Agent — hybrid search (semantic + BM25 + RRF)
      5. Reranker Agent  — Gemini pointwise reranking
      6. Guard          — confidence assessment
      7. Answer Agent   — grounded streaming response

    Returns a StreamingResponse of plain text.
    """
    question = request.question.strip()
    session_id = request.session_id

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        stream = await orchestrator.handle(question, session_id, request.collections)
        return StreamingResponse(stream, media_type="text/plain")
    except Exception as e:
        logger.error("Orchestrator error for session=%s: %s", session_id, e)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear conversation memory for a specific session."""
    chat_memory.clear(session_id)
    return {"message": f"Memory cleared for session '{session_id}'"}


@app.delete("/memory")
async def clear_all_memory():
    """Clear all conversation memory and semantic response cache."""
    chat_memory.clear_all()
    response_cache.clear()
    return {"message": "All memory and cache cleared"}
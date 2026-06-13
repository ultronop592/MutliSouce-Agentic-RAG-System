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
import hashlib
import shutil
import tempfile
import logging
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# ── Admin secret ─────────────────────────────────────────────────────────────────
# Loaded from the ADMIN_SECRET env var (set in Render dashboard).
# Required as X-Admin-Secret header on destructive admin endpoints.
ADMIN_SECRET = os.getenv("ADMIN_SECRET", "")

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
    doc_version: str | None = None      # MD5 fingerprint of active PDF
    source_filename: str | None = None  # filename of the active PDF (for Qdrant filter)


# ── Session → document version map ───────────────────────────────────────────
# Tracks the MD5 fingerprint of the last uploaded PDF per session.
_session_doc_version: dict[str, str] = {}

# ── Session → source filename map ──────────────────────────────────────────────
# Tracks the filename (basename) of the last uploaded PDF per session.
# Used by the retriever to filter Qdrant chunks to only the active PDF.
_session_source_file: dict[str, str] = {}



class UploadResponse(BaseModel):
    filename: str
    chunks_ingested: int
    collection: str
    doc_version: str = ""      # MD5 fingerprint — returned to frontend to use in chat requests
    source_filename: str = ""  # bare filename — returned so frontend can send it in chat requests
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


@app.delete("/collections/reset")
async def reset_all_collections(
    x_admin_secret: str | None = Header(default=None, alias="X-Admin-Secret"),
):
    """
    ⚠️  DESTRUCTIVE: Delete ALL points from every Qdrant collection.
    Requires the X-Admin-Secret header to match the ADMIN_SECRET env var.
    """    
    # ── Auth check ────────────────────────────────────────────────────────
    if not ADMIN_SECRET:
        raise HTTPException(status_code=503, detail="Admin secret not configured on server.")
    if x_admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Admin-Secret header.")
    from core.qdrant_client import qdrant
    from ingestion.ingestion import COLLECTIONS
    results = {}
    try:
        for collection in COLLECTIONS:
            try:
                # Delete ALL points — empty Filter() matches every point
                from qdrant_client.models import FilterSelector, Filter
                qdrant.delete(
                    collection_name=collection,
                    points_selector=FilterSelector(filter=Filter()),
                )
                # Rebuild (now empty) BM25 index
                bm25_manager.refresh(collection)
                results[collection] = "cleared"
                logger.info("Reset: cleared all points from '%s'", collection)
            except Exception as e:
                results[collection] = f"error: {str(e)}"
                logger.error("Reset: failed to clear '%s': %s", collection, e)

        # Clear server-side session state
        chat_memory.clear_all()
        response_cache.clear_all()
        _session_doc_version.clear()
        _session_source_file.clear()

        logger.info("Reset complete. Collections cleared: %s", results)
        return {
            "message": "All collections wiped. Re-upload your PDFs to populate with updated payload.",
            "collections": results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


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
    session_id: str | None = None,  # optional: frontend passes this to scope doc_version correctly
):
    """Upload a PDF and ingest it into Qdrant. Automatically refreshes BM25 index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        # Read file content ONCE — needed for both ingestion and MD5 hash
        content = await file.read()

        # ── Compute MD5 document fingerprint ──────────────────────────────────
        # This fingerprint acts as a "document version" that we tag on every
        # cache entry and memory turn. When the user uploads a different PDF,
        # the fingerprint changes → old cached answers are bypassed automatically.
        doc_version = hashlib.md5(content).hexdigest()
        logger.info("Upload: '%s' | doc_version=%s | session=%s", file.filename, doc_version, collection)

        # ── Write to temp file for ingestion ─────────────────────────────────
        with open(temp_path, "wb") as f:
            f.write(content)

        chunks_count = ingest_pdf(temp_path, collection)

        # ── Register doc_version in server-side map ──────────────────────────
        # Key by session_id (most accurate) when the frontend provides it,
        # AND by collection name (fallback for specific-tab uploads that don't
        # send session_id). This covers both Universal and specific-tab flows.
        bare_filename = os.path.basename(file.filename)
        if session_id:
            _session_doc_version[session_id] = doc_version
            _session_source_file[session_id] = bare_filename
            response_cache.set_doc_version(session_id, doc_version)
            logger.info("doc_version stored for session_id='%s' file='%s'", session_id, bare_filename)
        # Always also store by collection as a fallback
        _session_doc_version[collection] = doc_version
        _session_source_file[collection] = bare_filename
        response_cache.set_doc_version(collection, doc_version)

        logger.info(
            "Upload complete: '%s' | %d chunks | doc_version=%s",
            file.filename, chunks_count, doc_version,
        )

        return UploadResponse(
            filename=file.filename,
            chunks_ingested=chunks_count,
            collection=collection,
            doc_version=doc_version,
            source_filename=bare_filename,
            message=(
                f"Successfully ingested {chunks_count} chunks from '{file.filename}'. "
                f"BM25 index refreshed. Document version: {doc_version[:8]}..."
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

    # ── Resolve doc_version and source_filename ───────────────────────────────
    # Prefer values sent by the frontend (most accurate).
    # Fall back to the server-side map (populated on upload).
    doc_version = request.doc_version or _session_doc_version.get(session_id)
    source_filename = request.source_filename or _session_source_file.get(session_id)

    try:
        stream = await orchestrator.handle(
            question, session_id, request.collections,
            doc_version=doc_version,
            source_filename=source_filename,
        )
        return StreamingResponse(stream, media_type="text/plain")
    except Exception as e:
        logger.error("Orchestrator error for session=%s: %s", session_id, e)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear conversation memory AND session cache for a specific session.
    Called by the frontend 'New Chat' button so the next conversation
    starts completely fresh with no stale answers from the previous PDF."""
    chat_memory.clear(session_id)
    response_cache.clear(session_id)  # also clears doc_version for this session
    _session_doc_version.pop(session_id, None)
    _session_source_file.pop(session_id, None)
    return {"message": f"Memory and cache cleared for session '{session_id}'"}


@app.delete("/memory")
async def clear_all_memory():
    """Clear ALL conversation memory and ALL session caches."""
    chat_memory.clear_all()
    response_cache.clear_all()
    _session_doc_version.clear()
    _session_source_file.clear()
    return {"message": "All memory and cache cleared"}
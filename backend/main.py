import os
import uuid
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from retrieval.planner import planner
from retrieval.retriever import hybrid_retrieve
from retrieval.guard import assess_confidence
from retrieval.bm25_index import bm25_manager
from core.llm import llm
from core.embeddings import embeddings
from cache.cache import response_cache
from memory.memory import chat_memory
from ingestion.ingestion import ingest_pdf, ensure_collections, COLLECTIONS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure Qdrant collections exist and pre-load BM25 indexes on startup."""
    try:
        ensure_collections()
        print("Qdrant collections verified")
    except Exception as e:
        print(f" Qdrant connection warning: {e}")

    # Pre-load BM25 indexes from all existing Qdrant documents.
    # This ensures BM25 keyword search works from the very first request
    # even before any new uploads happen after server start.
    try:
        counts = bm25_manager.refresh_all(COLLECTIONS)
        for col, n in counts.items():
            if n > 0:
                print(f"BM25 index loaded: '{col}' — {n} docs")
        print("BM25 indexes ready")
    except Exception as e:
        print(f" BM25 index warning: {e}")

    yield


app = FastAPI(
    title="Multi-Source Agentic RAG API",
    description="RAG API with Qdrant, Gemini, and hybrid retrieval",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for Vercel + local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic Models ----------

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class UploadResponse(BaseModel):
    filename: str
    chunks_ingested: int
    collection: str
    message: str


# ---------- Endpoints ----------

@app.get("/")
async def root():
    return {
        "service": "Multi-Source Agentic RAG",
        "status": "running",
        "version": "2.0.0",
        "endpoints": ["/chat", "/upload", "/collections", "/memory"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/collections")
async def list_collections():
    """List all Qdrant collections with their point counts."""
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
                })
            except Exception:
                collection_list.append({
                    "name": col.name,
                    "points_count": 0,
                    "vectors_count": 0,
                })
        return {"collections": collection_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    collection: str = "research_papers",
):
    """Upload a PDF file and ingest it into a Qdrant collection."""
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
            message=f"Successfully ingested {chunks_count} chunks from {file.filename}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Chat endpoint with streaming, hybrid retrieval, hallucination guard,
    semantic cache, and conversation memory.
    """
    question = request.question.strip()
    session_id = request.session_id

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # ── Semantic cache lookup ────────────────────────────────────────────────
    # Embed the question for semantic cache lookup (single call, reused below).
    try:
        question_embedding = embeddings.embed_query(question)
        cached_answer = response_cache.get(question_embedding)
        if cached_answer:
            return {"answer": cached_answer, "cached": True}
    except Exception:
        question_embedding = None

    # ── Plan which collections to search ────────────────────────────────────
    selected = planner(question)

    # ── Hybrid retrieval — returns (docs, docs_with_scores) ─────────────────
    top_docs, docs_with_scores = await hybrid_retrieve(question, selected)

    # ── Hallucination Guard — assess retrieval confidence ───────────────────
    confidence_level, advisory_note = assess_confidence(docs_with_scores)

    # ── Build context string ─────────────────────────────────────────────────
    if top_docs:
        context = "\n\n---\n\n".join(top_docs)
    else:
        context = "No relevant documents found."

    # ── Get chat memory ──────────────────────────────────────────────────────
    memory = chat_memory.format_history(session_id)

    # ── Anti-Hallucination Prompt ────────────────────────────────────────────
    # This prompt is the primary defence against hallucination.
    # Key principles:
    #   1. Explicitly forbids adding any fact not present in the context.
    #   2. Injects the guard's advisory note so the LLM knows confidence level.
    #   3. Provides an explicit honest-fallback instruction.
    #   4. Removes any language that implicitly encourages inference/synthesis.
    prompt = f"""You are a precise, factual assistant. Your ONLY job is to answer
the user's question using the retrieved context below — nothing else.

{advisory_note}

STRICT GROUNDING RULES — violating these is your #1 failure mode:
1. ONLY use facts, numbers, names, dates, and claims that appear explicitly
   in the Retrieved Context section below.
2. Do NOT add any information from your general training knowledge, even if
   you are confident it is correct.
3. Do NOT infer, extrapolate, or "fill in" information that is absent from
   the context.
4. If the context does not contain enough information to answer the question
   fully, say: "I don't have enough information in my knowledge base to answer
   this accurately." — do NOT attempt a partial answer that invents details.
5. Do NOT use markdown headers (##, ###), bullet points, bold (**), or italic (*).
6. Do NOT include citation numbers like [1] or figure references like [Figure 1].
7. Write 2–3 plain sentences maximum. Be concise and direct.
8. Before writing each sentence, mentally verify: "Is this fact explicitly
   stated in the Retrieved Context?" If not, do not include it.

Previous Conversation:
{memory}

Retrieved Context:
{context}

User Question:
{question}

Answer (plain prose, strictly grounded in the context above):"""

    # ── Stream response ──────────────────────────────────────────────────────
    async def generate():
        full_response = ""
        try:
            response = llm.stream(prompt)
            for chunk in response:
                if chunk.content:
                    full_response += chunk.content
                    yield chunk.content
        except Exception as e:
            error_msg = f"\n\n[Error generating response: {str(e)}]"
            yield error_msg
            full_response += error_msg
        finally:
            # Update memory and semantic cache
            if full_response and not full_response.startswith("\n\n[Error"):
                chat_memory.update(session_id, question, full_response)
                # Only cache if we had a question embedding (non-error path)
                if question_embedding is not None:
                    response_cache.set(question_embedding, full_response)

    return StreamingResponse(generate(), media_type="text/plain")


@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear chat memory for a session."""
    chat_memory.clear(session_id)
    return {"message": f"Memory cleared for session {session_id}"}


@app.delete("/memory")
async def clear_all_memory():
    """Clear all chat memory and semantic cache."""
    chat_memory.clear_all()
    response_cache.clear()
    return {"message": "All memory and cache cleared"}
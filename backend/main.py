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
from core.llm import llm
from cache.cache import response_cache
from memory.memory import chat_memory
from ingestion.ingestion import ingest_pdf, ensure_collections


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure Qdrant collections exist on startup."""
    try:
        ensure_collections()
        print("✅ Qdrant collections verified")
    except Exception as e:
        print(f"⚠️  Qdrant connection warning: {e}")
    yield


app = FastAPI(
    title="Multi-Source Agentic RAG API",
    description="RAG API with Qdrant, Gemini, and hybrid retrieval",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001"],
    allow_credentials=True,
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
        "endpoints": ["/chat", "/upload", "/collections", "/memory"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/collections")
async def list_collections():
    """List all Qdrant collections with their point counts."""
    try:
        collections = []
        for c in ensure_collections():
            try:
                info = qdrant_info(c)
                collections.append(info)
            except Exception:
                collections.append({"name": c, "points_count": 0})
        
        # Also get current collections from Qdrant
        from core.qdrant_client import qdrant
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

    # Save to temp file
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Ingest into Qdrant
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
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with streaming response, hybrid retrieval, and memory."""
    question = request.question.strip()
    session_id = request.session_id

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Check cache
    if question in response_cache:
        return {"answer": response_cache[question], "cached": True}

    # Plan which collections to search
    selected = planner(question)

    # Hybrid retrieval (vector search + BM25 reranking)
    docs = await hybrid_retrieve(question, selected)

    # Build context
    context = "\n\n".join(docs) if docs else "No relevant documents found."

    # Get chat memory
    memory = chat_memory.format_history(session_id)

    # Build prompt
    prompt = f"""You are a multi-resource AI assistant. Answer the user's question based on the provided context.
If the context does not contain relevant information, say so honestly.
Be concise, accurate, and helpful.

Previous Conversation:
{memory}

Context from knowledge sources ({', '.join(selected)}):
{context}

Question:
{question}
"""

    # Stream response
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
            # Update memory and cache
            if full_response:
                chat_memory.update(session_id, question, full_response)
                response_cache[question] = full_response

    return StreamingResponse(generate(), media_type="text/plain")


@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear chat memory for a session."""
    chat_memory.clear(session_id)
    return {"message": f"Memory cleared for session {session_id}"}


@app.delete("/memory")
async def clear_all_memory():
    """Clear all chat memory."""
    chat_memory.clear_all()
    response_cache.clear()
    return {"message": "All memory and cache cleared"}
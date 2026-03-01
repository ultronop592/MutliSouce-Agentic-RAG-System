import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import VectorParams, Distance, PointStruct

from core.embeddings import embeddings, VECTOR_SIZE
from core.qdrant_client import qdrant


COLLECTIONS = [
    "research_papers",
    "knowledge_base",
    "code_docs",
    "faq_data",
]


def ensure_collections():
    """Create Qdrant collections if they don't exist. Recreate if vector size changed."""
    existing = {c.name: c for c in qdrant.get_collections().collections}
    for name in COLLECTIONS:
        if name in existing:
            # Check if vector size matches (handles migration from 384 to 768)
            try:
                info = qdrant.get_collection(name)
                if info.config.params.vectors.size != VECTOR_SIZE:
                    print(f"Recreating {name}: vector size changed to {VECTOR_SIZE}")
                    qdrant.delete_collection(name)
                    qdrant.create_collection(
                        collection_name=name,
                        vectors_config=VectorParams(
                            size=VECTOR_SIZE,
                            distance=Distance.COSINE,
                        ),
                    )
            except Exception:
                pass
        else:
            qdrant.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
    return list(existing.keys())


def ingest_pdf(file_path: str, collection: str = "research_papers"):
    """
    Load a PDF, split into chunks, embed, and upsert into Qdrant.
    Returns the number of chunks ingested.
    """
    # Ensure collections exist
    ensure_collections()

    # Load PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(data)

    if not chunks:
        return 0

    # Embed all chunks
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)

    # Build Qdrant points
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vectors[i],
            payload={
                "text": chunks[i].page_content,
                "page": chunks[i].metadata.get("page", 0),
                "source_file": file_path,
                "collection": collection,
            },
        )
        for i in range(len(chunks))
    ]

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        qdrant.upsert(collection_name=collection, points=batch)

    return len(points)

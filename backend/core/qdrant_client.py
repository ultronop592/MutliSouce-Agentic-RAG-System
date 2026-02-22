from qdrant_client import QdrantClient
from core.config import QDRANT_URL, QDRANT_API_KEY

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
    prefer_grpc=False,
)
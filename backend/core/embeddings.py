from langchain_google_genai import GoogleGenerativeAIEmbeddings
from core.config import GEMINI_API_KEY

# Use Google's Embedding API instead of local sentence-transformers.
# This uses ZERO local memory (no PyTorch needed) — critical for Render free tier (512MB).
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY,
)

VECTOR_SIZE = 768  # Dimension of Google embedding-001
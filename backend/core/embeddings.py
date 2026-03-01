from langchain_google_genai import GoogleGenerativeAIEmbeddings
from core.config import GEMINI_API_KEY

# Use Google's Gemini Embedding API — zero local memory needed.
# gemini-embedding-001 is the current model (replaces text-embedding-004).
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
)

VECTOR_SIZE = 3072  # Dimension of gemini-embedding-001
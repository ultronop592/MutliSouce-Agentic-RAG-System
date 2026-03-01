VECTOR_SIZE = 384  # Dimension of all-MiniLM-L6-v2

# Lazy-load the embedding model AND the heavy import (PyTorch/sentence-transformers)
# so uvicorn can bind the port immediately on Render.
_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


class _EmbeddingsProxy:
    """Proxy that delays model loading until first actual use."""
    def __getattr__(self, name):
        return getattr(get_embeddings(), name)


embeddings = _EmbeddingsProxy()
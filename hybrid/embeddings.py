from sentence_transformers import SentenceTransformer
import numpy as np

# 384 dimensiones:
_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed(text: str) -> np.ndarray:
    v = _model.encode([text], normalize_embeddings=True)[0]  # vector unitario
    return v.astype("float32")
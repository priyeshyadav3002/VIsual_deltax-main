import io
import json
import requests
import numpy as np
from PIL import Image
from typing import Optional

def fetch_image_bytes(url: str, timeout: int = 10) -> Optional[bytes]:
    """Download image bytes from URL."""
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print("fetch_image_bytes error:", e)
        return None


def load_image_from_bytes(b: bytes) -> Image.Image:
    """Convert bytes to PIL image."""
    return Image.open(io.BytesIO(b)).convert("RGB")

def parse_embedding_field(embedding_field):
    """Parse stored embedding (list or JSON string) into numpy array."""
    if embedding_field is None:
        return None
    if isinstance(embedding_field, list):
        return np.array(embedding_field, dtype=np.float32)
    if isinstance(embedding_field, str):
        try:
            arr = json.loads(embedding_field)
            return np.array(arr, dtype=np.float32)
        except Exception:
            return None
    return None

def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def top_k_similar(query_vec: np.ndarray, candidate_vecs, k: int = 10):
    """Compute cosine similarity between query and candidates."""
    q = normalize_vector(query_vec)
    mat = np.vstack(candidate_vecs)
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)
    sims = mat.dot(q)
    top_idx = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top_idx]

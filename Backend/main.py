from dotenv import load_dotenv
from pathlib import Path
import torch
import clip
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional

from backend.utils import (
    fetch_image_bytes,
    load_image_from_bytes,
    parse_embedding_field,
    top_k_similar,
    normalize_vector,
)
from backend.database import supabase

# Load .env explicitly
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# Load CLIP model
device = "cpu"  # HF Spaces CPU
_model, _preprocess = clip.load("ViT-B/32", device=device)
_model.eval()

# FastAPI app
app = FastAPI(title="Visual Product Matcher")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---

@app.get("/health")
async def health():
    return {"status": "ok"}

async def embed_image_pil(img):
    """Convert PIL image into normalized CLIP embedding"""
    image_tensor = _preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = _model.encode_image(image_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    vec = features[0].cpu().numpy().astype("float32")
    return normalize_vector(vec)

@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(None),
    image_url: Optional[str] = Form(None),
    top_k: int = Form(8),
):
    if file is None and not image_url:
        raise HTTPException(status_code=400, detail="Provide file or image_url")

    if file:
        image_bytes = await file.read()
    else:
        image_bytes = fetch_image_bytes(image_url)
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Could not fetch image")

    try:
        img = load_image_from_bytes(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    query_vec = await embed_image_pil(img)

    # Get embeddings from DB
    resp = supabase.table("products").select(
        "id,name,category,image_url,embedding"
    ).execute()
    items = resp.data or []

    candidate_vecs, candidates = [], []
    for it in items:
        emb = parse_embedding_field(it.get("embedding"))
        if emb is not None:
            candidates.append(it)
            candidate_vecs.append(emb)

    if not candidate_vecs:
        raise HTTPException(status_code=500, detail="No embeddings available in database")

    # Similarity ranking
    top = top_k_similar(query_vec, candidate_vecs, k=top_k)
    results = []
    for idx, sim in top:
        meta = candidates[idx]
        results.append(
            {
                "id": meta["id"],
                "name": meta.get("name"),
                "category": meta.get("category"),
                "image_url": meta.get("image_url"),
                "similarity": sim,
            }
        )

    return {"results": results}

# --- Serve frontend (index.html, styles.css, script.js) ---
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

import clip
import torch

_model = None
_preprocess = None
_device = None

def get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device

def load_clip(model_name: str = "ViT-B/32", local_path: str = None):
    """
    Lazy-loads the CLIP model and processor using clip-by-openai.
    """
    global _model, _preprocess
    if _model is None or _preprocess is None:
        print(f"Loading CLIP model {model_name} locally...")
        _model, _preprocess = clip.load(model_name, device=get_device())
        _model.eval()
    return _model, _preprocess

def get_image_embedding(image):
    """
    image: PIL.Image object
    returns: torch tensor embedding
    """
    model, preprocess = load_clip()
    image_tensor = preprocess(image).unsqueeze(0).to(get_device())
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
    return embedding
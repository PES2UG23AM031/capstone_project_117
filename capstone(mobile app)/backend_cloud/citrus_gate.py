import os
import torch, torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

_model = None
_thr = 0.4
_dev = "cpu"
_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Three-outcome policy (tuned on the dev exams; adjust after a larger test).
ACCEPT_AT = 0.60      # p >= this        -> accept
REJECT_BELOW = 0.30   # p <  this        -> reject; in between -> retake


def load_gate(path, device="cpu"):
    global _model, _thr, _dev
    if not os.path.exists(path):
        raise FileNotFoundError(f"Gate model not found at: {path}")
    _dev = device
    ckpt = torch.load(path, map_location=device)
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    m.load_state_dict(ckpt["state_dict"])
    _model = m.to(device).eval()
    _thr = float(ckpt.get("threshold", 0.4))
    print(f"Citrus gate loaded: classes={ckpt['classes']} threshold={_thr} "
          f"accept>={ACCEPT_AT} retake>={REJECT_BELOW}")


def _score(image):
    """image: file path, file-like object, or PIL.Image. Returns P(citrus)."""
    if _model is None:
        raise RuntimeError("Citrus gate not loaded. Call load_gate(path) at startup.")
    im = image if isinstance(image, Image.Image) else Image.open(image)
    im = ImageOps.exif_transpose(im).convert("RGB")
    x = _tf(im).unsqueeze(0).to(_dev)
    with torch.no_grad():
        return torch.softmax(_model(x), dim=1)[0, 0].item()


def check_citrus(image_path):
    """Original API. Returns (is_citrus_leaf, p_citrus) at the checkpoint threshold. Class index 0 = citrus."""
    p = _score(image_path)
    return p >= _thr, p


def check_citrus_decision(image):
    """Returns (decision, p_citrus) where decision is 'accept', 'retake' or 'reject'."""
    p = _score(image)
    if p >= ACCEPT_AT:
        return "accept", p
    if p >= REJECT_BELOW:
        return "retake", p
    return "reject", p
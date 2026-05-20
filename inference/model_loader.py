import copy
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms

load_dotenv()

EMOTION_LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]

_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def _ensure_fer_vit_on_path() -> None:
    fer_vit_path = os.getenv("FER_VIT_PATH", "../fer-vit")
    src = str(Path(fer_vit_path).resolve() / "src")
    if not Path(src).exists():
        raise RuntimeError(f"FER_VIT_PATH src directory not found: {src}")
    if src not in sys.path:
        sys.path.insert(0, src)


def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(config)
    m = cfg.get("model", {})
    m["pretrained"] = False
    m.pop("dinov3_checkpoint", None)
    m.pop("farl_checkpoint", None)
    m["freeze_backbone"] = False
    return cfg


def load_model(checkpoint_path: str) -> Tuple[nn.Module, Dict[str, Any]]:
    _ensure_fer_vit_on_path()

    from vit.models.vit_model import create_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if config is None:
        raise ValueError(f"Checkpoint has no embedded config: {checkpoint_path}")

    config = _sanitize_config(config)
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config


def run_inference(
    model: nn.Module,
    face_rgb: np.ndarray,
    device: torch.device,
) -> Dict[str, int]:
    image = Image.fromarray(face_rgb)
    tensor = _INFER_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)

    predicted = logits.argmax(dim=2).squeeze(0).cpu().tolist()
    return {label: int(cls) for label, cls in zip(EMOTION_LABELS, predicted)}

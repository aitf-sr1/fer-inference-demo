import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from dotenv import load_dotenv

from .face_detector import create_detector, detect_and_crop
from .model_loader import load_model, run_inference

load_dotenv()

_ASSETS_DIR = Path(__file__).parent.parent / "assets"
_DEFAULT_MEDIAPIPE = str(_ASSETS_DIR / "blaze_face_short_range.tflite")


class InferencePipeline:
    def __init__(self) -> None:
        mediapipe_path = Path(
            os.getenv("MEDIAPIPE_MODEL_PATH", _DEFAULT_MEDIAPIPE)
        ).resolve()
        self._models_dir = Path(os.getenv("MODELS_DIR", "./models")).resolve()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._detector = create_detector(str(mediapipe_path))
        self._model: Optional[Any] = None
        self._current_model_name: Optional[str] = None
        self._num_classes: int = 4

    def list_models(self) -> List[str]:
        return sorted(p.name for p in self._models_dir.glob("*.pth"))

    def load_model(self, model_name: str) -> None:
        if model_name == self._current_model_name:
            return
        path = self._models_dir / model_name
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self._model, config = load_model(str(path))
        self._model.to(self._device)
        self._current_model_name = model_name
        self._num_classes = config.get("model", {}).get("num_classes", 4)

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model_name

    def infer(self, image_rgb: np.ndarray) -> Dict[str, Any]:
        if self._model is None:
            raise RuntimeError("No model loaded.")
        face = detect_and_crop(image_rgb, self._detector)
        if face is None:
            return {"face_detected": False, "emotions": None, "num_classes": self._num_classes}
        emotions = run_inference(self._model, face, self._device)
        return {"face_detected": True, "emotions": emotions, "num_classes": self._num_classes}

    def infer_base64(self, b64_image: str) -> Dict[str, Any]:
        if "," in b64_image:
            b64_image = b64_image.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_image)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Could not decode image.")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.infer(img_rgb)

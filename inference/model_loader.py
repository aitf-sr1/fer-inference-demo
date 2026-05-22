from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort

EMOTION_LABELS = ["Boredom", "Engagement", "Confusion", "Frustration"]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model(
    checkpoint_path: str, providers: List[str]
) -> Tuple[ort.InferenceSession, int]:
    session = ort.InferenceSession(checkpoint_path, providers=providers)
    output_shape = session.get_outputs()[0].shape
    dim2 = output_shape[2] if len(output_shape) >= 3 else None
    num_classes = int(dim2) if isinstance(dim2, int) else 4
    return session, num_classes


def run_inference(
    session: ort.InferenceSession,
    face_rgb: np.ndarray,
) -> Dict[str, int]:
    img = face_rgb.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    img = img.transpose(2, 0, 1)[np.newaxis]

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: img})[0]

    predicted = logits.argmax(axis=2).squeeze(0).tolist()
    return {label: int(cls) for label, cls in zip(EMOTION_LABELS, predicted)}

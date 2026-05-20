from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

OUTPUT_SIZE = 224


def create_detector(model_path: str) -> vision.FaceDetector:
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(options)


def detect_and_crop(
    img_rgb: np.ndarray,
    detector: vision.FaceDetector,
) -> Optional[np.ndarray]:
    h, w = img_rgb.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = detector.detect(mp_image)

    if not result.detections:
        return None

    bbox = result.detections[0].bounding_box
    x_min, y_min = bbox.origin_x, bbox.origin_y
    box_w, box_h = bbox.width, bbox.height

    pad_x = int(box_w * 0.05)
    pad_top = int(box_h * 0.20)
    pad_bottom = int(box_h * 0.10)

    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_top)
    x2 = min(w, x_min + box_w + pad_x)
    y2 = min(h, y_min + box_h + pad_bottom)

    cropped = img_rgb[y1:y2, x1:x2]
    if cropped.size == 0:
        return None

    return cv2.resize(cropped, (OUTPUT_SIZE, OUTPUT_SIZE))

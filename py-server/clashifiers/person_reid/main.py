"""
Person Re-Identification (Body Appearance Embedding)
-----------------------------------------------------
Model: OSNet (via torchreid), pretrained on Market-1501 / MSMT17.
Input: BGR numpy array, a *cropped* person bounding box (not the full frame).
Output: L2-normalised embedding vector (512-d for osnet_x1_0).

This is the "fallback" identity signal used when no confident face match
is available for a detected person (e.g. face not visible / too small /
person facing away from camera).

Install:
    pip install torchreid torch torchvision

Usage:
    reid = PersonReIdentifier(device="cpu")
    vec = reid.embed(person_crop_bgr)          # -> np.ndarray shape (512,)
    sim = PersonReIdentifier.cosine_similarity(vec_a, vec_b)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger("person_reid")

_MODEL_NAME = "osnet_x1_0"
_INPUT_SIZE = (256, 128)  # (height, width) — standard Re-ID input size
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class ReIdEmbeddingResult:
    embedding: np.ndarray  # shape (512,), L2-normalised
    model: str = _MODEL_NAME


class PersonReIdentifier:
    """
    Wraps a torchreid OSNet model to turn a cropped person image into a
    fixed-length appearance embedding usable for cosine-similarity matching.
    """

    def __init__(self, device: str = "cpu", model_name: str = _MODEL_NAME):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._torch = None
        self._load_model()

    def _load_model(self):
        try:
            import torch
            import torchreid

            self._torch = torch

            model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=1,  # unused at inference time, embeddings only
                pretrained=True,
            )
            model.eval()
            model.to(self.device)
            self._model = model
            logger.info(f"PersonReIdentifier loaded '{self.model_name}' on {self.device}")
        except Exception as e:
            logger.warning(
                f"PersonReIdentifier: failed to load torchreid model "
                f"({e}). Body Re-ID matching will be disabled."
            )
            self._model = None

    @property
    def is_available(self) -> bool:
        return self._model is not None

    def _preprocess(self, crop_bgr: np.ndarray):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (_INPUT_SIZE[1], _INPUT_SIZE[0]), interpolation=cv2.INTER_LINEAR)
        normed = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        chw = np.transpose(normed, (2, 0, 1))  # HWC -> CHW
        tensor = self._torch.from_numpy(chw).unsqueeze(0).float().to(self.device)
        return tensor

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        crop_bgr: a cropped person image (already bbox-cropped from the frame).
        Returns an L2-normalised embedding, or None if the model isn't available
        or the crop is invalid.
        """
        if not self.is_available:
            return None
        if crop_bgr is None or crop_bgr.size == 0:
            return None

        tensor = self._preprocess(crop_bgr)
        with self._torch.no_grad():
            features = self._model(tensor)
            if isinstance(features, (tuple, list)):
                features = features[0]
            vec = features.cpu().numpy().reshape(-1).astype(np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
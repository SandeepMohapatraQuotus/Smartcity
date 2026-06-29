"""
Zero-DCE++ Night Enhancement Service
--------------------------------------
Path : py-server/services/zero_dce.py

Wraps the Zero-DCE++ model (from Zero-DCE_extension/) into a clean,
pipeline-friendly interface that:

  1.  Loads the pretrained model once  (Epoch99.pth)
  2.  Accepts a BGR numpy array  (OpenCV convention)
  3.  Returns an enhanced BGR numpy array
  4.  Works on both CPU and GPU transparently
  5.  Falls back to CLAHE if model weights are missing

Usage inside pipeline.py:
    from services.zero_dce import ZeroDCEEnhancer
    enhancer = ZeroDCEEnhancer()                       # loads model
    enhanced = enhancer.enhance(frame)                  # BGR in → BGR out

The original Zero-DCE++ repo lives at:
    Zero-DCE_extension/Zero-DCE++/
Its directory name contains "++" so it can't be imported normally —
we add it to sys.path and import the `model` module directly.
"""

import os
import sys
import cv2
import time
import numpy as np
import torch

# ─── Resolve the Zero-DCE++ module path ──────────────────────────────────────
# The folder is called "Zero-DCE++" which is not a valid Python identifier,
# so we add its parent to sys.path and import the `model` module by name.

_HERE        = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_HERE)                                   # py-server/
_ZDCE_DIR    = os.path.join(_PROJECT_DIR, "Zero-DCE_extension", "Zero-DCE++")
_WEIGHTS     = os.path.join(_ZDCE_DIR, "snapshots_Zero_DCE++", "Epoch99.pth")

if _ZDCE_DIR not in sys.path:
    sys.path.insert(0, _ZDCE_DIR)

# Now we can import the model module from Zero-DCE++/model.py
from model import enhance_net_nopool  # type: ignore[import-untyped]


# ─── CLAHE Fallback ──────────────────────────────────────────────────────────

def _clahe_enhance(frame: np.ndarray) -> np.ndarray:
    """
    Baseline CLAHE enhancement — used as fallback when Zero-DCE++
    weights are not available.
    """
    lab     = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ─── Enhancer ─────────────────────────────────────────────────────────────────

class ZeroDCEEnhancer:
    """
    Production-ready wrapper around Zero-DCE++ for low-light enhancement.

    Constructor args:
        weights_path : path to .pth file  (default: snapshots_Zero_DCE++/Epoch99.pth)
        scale_factor : downscale factor for inference speed
                       1  = full resolution  (best quality, slower)
                       12 = paper default    (fast, used in lowlight_test.py)
        device       : "auto" | "cpu" | "cuda" | "cuda:0"

    If the weights file is missing the enhancer gracefully falls back to CLAHE
    and prints a warning — the pipeline never crashes.
    """

    def __init__(
        self,
        weights_path : str = _WEIGHTS,
        scale_factor : int = 1,
        device       : str = "auto",
    ):
        self.scale_factor = scale_factor
        self.device       = self._resolve_device(device)
        self._model       = None       # lazy-loaded on first call if needed
        self._fallback    = False      # True → weights missing, use CLAHE

        self._load_model(weights_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance a single low-light BGR frame.

        Args:
            frame : BGR numpy array from OpenCV  (any resolution)

        Returns:
            Enhanced BGR numpy array  (same resolution as input)
        """
        # ── Save original dimensions ──────────────────────────────────────────
        # Zero-DCE++ pads the frame to a scale_factor multiple before running
        # the CNN, then crops it back.  Floating-point rounding in that
        # pad/crop cycle can leave the output 1-2 px off from the input, which
        # causes ByteTrack's GMC (Lucas-Kanade pyramid) to throw an assertion
        # error when the previous and current frame sizes don't match.
        orig_h, orig_w = frame.shape[:2]

        # ── Run enhancement ───────────────────────────────────────────────────
        if self._fallback:
            result = _clahe_enhance(frame)
        else:
            result = self._zero_dce_enhance(frame)

        # ── Guarantee exact size match ────────────────────────────────────────
        # Applies even on the CLAHE path: if the input somehow had an odd
        # dimension that CLAHE's internal tiling rounded, we still get back
        # the exact (orig_w, orig_h) the caller gave us.
        if result.shape[:2] != (orig_h, orig_w):
            result = cv2.resize(
                result,
                (orig_w, orig_h),           # cv2.resize takes (width, height)
                interpolation=cv2.INTER_LINEAR,
            )

        return result

    @property
    def method(self) -> str:
        """Return which enhancement backend is active."""
        return "clahe" if self._fallback else "zero_dce++"

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self, weights_path: str):
        """Load the Zero-DCE++ network and pretrained weights."""
        if not os.path.isfile(weights_path):
            print(f"[ZeroDCE] WARNING: weights not found at {weights_path}")
            print(f"[ZeroDCE] Falling back to CLAHE enhancement.")
            self._fallback = True
            return

        try:
            self._model = enhance_net_nopool(self.scale_factor).to(self.device)
            state_dict  = torch.load(weights_path, map_location=self.device)
            self._model.load_state_dict(state_dict)
            self._model.eval()
            print(f"[ZeroDCE] Model loaded — scale_factor={self.scale_factor}, "
                  f"device={self.device}, weights={os.path.basename(weights_path)}")
        except Exception as e:
            print(f"[ZeroDCE] Failed to load model: {e}")
            print(f"[ZeroDCE] Falling back to CLAHE enhancement.")
            self._fallback = True
            self._model    = None

    def _zero_dce_enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Core Zero-DCE++ inference.

        Pipeline:
          BGR (uint8) → RGB (float32 [0,1]) → tensor → model → numpy → BGR (uint8)

        Height and width are padded to be divisible by scale_factor,
        then cropped back to the original size after enhancement.
        """
        orig_h, orig_w = frame.shape[:2]

        # BGR → RGB, normalise to [0, 1]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = rgb.astype(np.float32) / 255.0

        # Pad dimensions to be divisible by scale_factor
        sf = self.scale_factor
        pad_h = (sf - (orig_h % sf)) % sf
        pad_w = (sf - (orig_w % sf)) % sf
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        # numpy (H, W, 3) → tensor (1, 3, H, W)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            enhanced_tensor, _ = self._model(tensor)

        # tensor (1, 3, H, W) → numpy (H, W, 3) RGB → BGR uint8
        enhanced = enhanced_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)

        # Crop back to original size
        enhanced = enhanced[:orig_h, :orig_w, :]

        # RGB → BGR
        return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

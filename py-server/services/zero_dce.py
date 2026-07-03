"""
Night Enhancement Service — Chained Pipeline
    Gamma Correction  →  CLAHE  →  Zero-DCE++
--------------------------------------------------------------------
Path : py-server/services/zero_dce.py

Runs every enhancement stage on the frame IN SEQUENCE, each stage
operating on the output of the previous one:

    raw frame
        │
        ▼
    1. Gamma Correction   — fixes GLOBAL exposure (adaptive: gamma is
                             computed per-frame from its mean luminance,
                             so a very dark frame gets brightened more
                             than a mildly dark one).
        │
        ▼
    2. CLAHE               — fixes LOCAL contrast (recovers detail in
                             shadows/highlights left flat by step 1;
                             operates on the L channel of LAB so colour
                             is preserved).
        │
        ▼
    3. Zero-DCE++ (CNN)    — final deep-learned refinement pass. Skipped
                             automatically (with a one-time warning) if
                             its weights aren't available — the first two
                             classical stages still run either way, so the
                             pipeline never crashes for lack of weights.
        │
        ▼
    enhanced frame

Any stage can be switched off individually via the constructor flags
below if you only want a subset of the chain.

Usage inside pipeline.py:
    from services.zero_dce import ZeroDCEEnhancer
    enhancer = ZeroDCEEnhancer()                # all 3 stages on by default
    enhanced = enhancer.enhance(frame)          # BGR in → BGR out, chained
    enhancer.last_stages_applied                # e.g. ["gamma", "clahe", "zero_dce"]
    enhancer.last_gamma                         # gamma value used on this frame

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

from services.classical_enhance import gamma_correction, estimate_gamma, clahe_enhance

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


# ─── Enhancer ─────────────────────────────────────────────────────────────────

class ZeroDCEEnhancer:
    """
    Chained low-light enhancer: Gamma Correction → CLAHE → Zero-DCE++.

    Every frame passed to `enhance()` flows through whichever stages are
    enabled, each stage transforming the previous stage's output (not the
    original frame independently three times).

    Constructor args:
        weights_path          : path to Zero-DCE++ .pth file.
        scale_factor           : Zero-DCE++ internal downscale factor
                                  (1 = full res/slow/best, 12 = paper default).
        device                 : "auto" | "cpu" | "cuda" | "cuda:0".

        enable_gamma           : run adaptive Gamma Correction stage.  (default True)
        enable_clahe            : run CLAHE stage.                      (default True)
        enable_zero_dce         : run Zero-DCE++ CNN stage.             (default True)

        gamma_target_mean       : target mean luminance (0-255) the adaptive
                                   gamma stage aims for.
        clahe_clip_limit        : CLAHE contrast clip threshold.
        clahe_tile_grid_size    : CLAHE tile grid size.

    If `enable_zero_dce=True` but the weights file is missing/broken, that
    stage is skipped automatically (warning printed once at startup) — the
    gamma and CLAHE stages still run, so the pipeline never crashes for
    lack of model weights.
    """

    def __init__(
        self,
        weights_path          : str   = _WEIGHTS,
        scale_factor            : int   = 12,
        device                  : str   = "cpu",
        enable_gamma             : bool  = True,
        enable_clahe             : bool  = True,
        enable_zero_dce          : bool  = True,
        gamma_target_mean        : float = 128.0,
        clahe_clip_limit         : float = 3.0,
        clahe_tile_grid_size     : tuple = (8, 8),
    ):
        self.scale_factor = scale_factor
        self.device       = self._resolve_device(device)
        self._model       = None

        self.enable_gamma     = enable_gamma
        self.enable_clahe     = enable_clahe
        self.enable_zero_dce  = enable_zero_dce   # requested state
        self._zero_dce_ready  = False              # actual state (weights loaded ok?)

        self.gamma_target_mean    = gamma_target_mean
        self.clahe_clip_limit     = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size

        # Per-frame diagnostics, updated on every enhance() call
        self.last_gamma           : float | None = None
        self.last_stages_applied  : list[str]     = []

        if self.enable_zero_dce:
            self._load_model(weights_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Run the frame through every enabled stage, in order:
            Gamma Correction → CLAHE → Zero-DCE++

        Each stage's output becomes the next stage's input.

        Args:
            frame : BGR numpy array from OpenCV  (any resolution)

        Returns:
            Enhanced BGR numpy array  (same resolution as input)
        """
        orig_h, orig_w = frame.shape[:2]
        working = frame
        stages_applied: list[str] = []
        self.last_gamma = None

        # ── Stage 1: Gamma Correction (global exposure) ────────────────────────
        if self.enable_gamma:
            gamma = estimate_gamma(working, target_mean=self.gamma_target_mean)
            working = gamma_correction(working, gamma=gamma)
            self.last_gamma = round(gamma, 3)
            stages_applied.append("gamma")

        # ── Stage 2: CLAHE (local contrast) ─────────────────────────────────────
        if self.enable_clahe:
            working = clahe_enhance(
                working,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_tile_grid_size,
            )
            stages_applied.append("clahe")

        # ── Stage 3: Zero-DCE++ (deep refinement) ────────────────────────────────
        # NOTE: Zero-DCE++ pads to a scale_factor multiple internally, then crops
        # back — floating point rounding in that pad/crop cycle can leave the
        # output 1-2px off from its input, which is why every stage re-checks
        # size at the end regardless of which stages ran.
        if self.enable_zero_dce and self._zero_dce_ready:
            working = self._zero_dce_enhance(working)
            stages_applied.append("zero_dce")

        self.last_stages_applied = stages_applied

        # ── Guarantee exact size match on the final output ─────────────────────
        if working.shape[:2] != (orig_h, orig_w):
            working = cv2.resize(
                working,
                (orig_w, orig_h),           # cv2.resize takes (width, height)
                interpolation=cv2.INTER_LINEAR,
            )

        return working

    @property
    def method(self) -> str:
        """
        Human-readable summary of what actually ran on the last frame, e.g.
        "gamma[1.62]+clahe+zero_dce" or "gamma[2.1]+clahe" if zero_dce weights
        are unavailable. Kept as `method` (not `stages_applied`) for backward
        compatibility with callers that read `enhancer.method`.
        """
        if not self.last_stages_applied:
            return "none"
        parts = []
        for stage in self.last_stages_applied:
            if stage == "gamma" and self.last_gamma is not None:
                parts.append(f"gamma[{self.last_gamma}]")
            else:
                parts.append(stage)
        return "+".join(parts)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_model(self, weights_path: str):
        """Load the Zero-DCE++ network and pretrained weights."""
        if not os.path.isfile(weights_path):
            print(f"[ZeroDCE] WARNING: weights not found at {weights_path}")
            print(f"[ZeroDCE] Zero-DCE++ stage will be SKIPPED — "
                  f"gamma/CLAHE stages still run.")
            self._zero_dce_ready = False
            return

        try:
            self._model = enhance_net_nopool(self.scale_factor).to(self.device)
            state_dict  = torch.load(weights_path, map_location=self.device)
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self._zero_dce_ready = True
            print(f"[ZeroDCE] Model loaded — scale_factor={self.scale_factor}, "
                  f"device={self.device}, weights={os.path.basename(weights_path)}")
        except Exception as e:
            print(f"[ZeroDCE] Failed to load model: {e}")
            print(f"[ZeroDCE] Zero-DCE++ stage will be SKIPPED — "
                  f"gamma/CLAHE stages still run.")
            self._zero_dce_ready = False
            self._model = None

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
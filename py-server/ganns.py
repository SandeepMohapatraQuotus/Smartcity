"""
services/sr_enhance.py — SRGANEnhancer

Super-resolution preprocessing stage, run BEFORE day/night classification
and BEFORE any detector. Intended to fix recall loss on low-res source
video (e.g. 144p) where person/face detectors have too few pixels to work
with, before any night-enhancement or detection stage ever sees the frame.

Design notes:
- Uses a *compact* SR network (SRVGG-style, e.g. realesr-general-x4v3),
  NOT full RRDBNet ESRGAN — the deep RRDB variant is too slow for
  per-frame CPU video (multi-second latency at 4x). The compact net is
  still GAN-trained but ~10-20x cheaper to run.
- Only triggers when the frame is actually low-res (below `min_side_trigger`)
  so already-fine 240p+/480p+ streams skip this stage entirely.
- Falls back to bicubic + light unsharp mask if the model/weights aren't
  available, so the pipeline never hard-fails without the SR dependency
  installed (same pattern as SCIEnhancer's CLAHE-only fallback).
- API mirrors SCIEnhancer / ZeroDCEEnhancer: .enhance(frame_bgr) -> frame_bgr,
  plus .method / .last_scale_applied / .last_triggered for observability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from realesrgan import RealESRGANer
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False


@dataclass
class SRResult:
    frame: np.ndarray
    triggered: bool
    scale_applied: float
    method: str


class SRGANEnhancer:
    """
    Conditional super-resolution preprocessing stage.

    Constructor params (mirrors SCIEnhancer's style):
        weights_path        — path to compact SR .pth checkpoint
                               (defaults to weights/realesr-general-x4v3.pth)
        device               — 'cpu' or 'cuda'
        scale                — upscale factor the model was trained for (usually 4)
        min_side_trigger     — only run SR if min(h, w) of the frame is below this.
                                Frames at or above this resolution pass through untouched.
        target_min_side      — after SR, downscale back so the shorter side is at most
                                this value (avoids feeding detectors an unnecessarily
                                huge frame and blowing up downstream latency).
        tile                 — tile size for RealESRGANer (limits peak memory on CPU;
                                0 disables tiling)
        n_threads            — torch CPU thread count
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        scale: int = 4,
        min_side_trigger: int = 260,
        target_min_side: int = 480,
        tile: int = 200,
        n_threads: int = 4,
    ):
        self.device = device
        self.scale = scale
        self.min_side_trigger = min_side_trigger
        self.target_min_side = target_min_side
        self.tile = tile
        self.method = "bicubic_fallback"
        self.last_triggered = False
        self.last_scale_applied = 1.0

        self._upsampler = None

        if not _SR_AVAILABLE:
            logger.warning(
                "realesrgan/basicsr/torch not installed — SRGANEnhancer will "
                "fall back to bicubic + unsharp for low-res frames."
            )
            return

        weights_path = weights_path or "weights/realesr-general-x4v3.pth"
        try:
            if device == "cpu":
                torch.set_num_threads(n_threads)

            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=32, upscale=scale, act_type="prelu",
            )
            self._upsampler = RealESRGANer(
                scale=scale,
                model_path=weights_path,
                model=model,
                tile=tile,
                tile_pad=10,
                pre_pad=0,
                half=(device == "cuda"),
                device=device,
            )
            self.method = "srgan_compact"
        except Exception as e:
            logger.warning("Failed to load SR weights (%s) — using bicubic fallback.", e)
            self._upsampler = None

    def _bicubic_fallback(self, frame_bgr: np.ndarray, scale: float) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        up = cv2.resize(
            frame_bgr, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
        blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.0)
        sharpened = cv2.addWeighted(up, 1.5, blur, -0.5, 0)
        return sharpened

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Conditionally super-resolve a frame. Returns the original frame
        unchanged if it's already above min_side_trigger.
        """
        h, w = frame_bgr.shape[:2]
        min_side = min(h, w)

        if min_side >= self.min_side_trigger:
            self.last_triggered = False
            self.last_scale_applied = 1.0
            return frame_bgr

        self.last_triggered = True

        if self._upsampler is not None:
            try:
                output, _ = self._upsampler.enhance(frame_bgr, outscale=self.scale)
            except Exception as e:
                logger.warning("SR inference failed (%s) — falling back to bicubic.", e)
                output = self._bicubic_fallback(frame_bgr, self.scale)
        else:
            output = self._bicubic_fallback(frame_bgr, self.scale)

        # Cap output size: don't hand detectors a needlessly huge frame.
        oh, ow = output.shape[:2]
        out_min_side = min(oh, ow)
        if out_min_side > self.target_min_side:
            cap_scale = self.target_min_side / out_min_side
            output = cv2.resize(
                output, (int(ow * cap_scale), int(oh * cap_scale)),
                interpolation=cv2.INTER_AREA,
            )

        self.last_scale_applied = min(output.shape[0] / h, output.shape[1] / w)
        return output
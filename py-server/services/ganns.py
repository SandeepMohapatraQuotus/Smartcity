"""
services/sr_enhance.py — SRGANEnhancer (standalone, no basicsr/realesrgan dependency)

Why standalone: basicsr's realesrgan_dataset.py imports
`torchvision.transforms.functional_tensor`, which was removed in modern
torchvision (>=0.17, folded into `functional`). basicsr hasn't been updated
for it, so `import basicsr` throws ModuleNotFoundError on any recent
torch/torchvision install — and since SRGANEnhancer originally caught that
as a bare ImportError, it silently fell back to bicubic with no visible
failure at request time.

Fix: we only ever needed the SRVGGNetCompact architecture (7 conv layers)
to run inference on the realesr-general-x4v3.pth checkpoint — not
basicsr's training/dataset machinery. Defining it inline removes the
dependency (and this whole class of future breakage) entirely.

Checkpoint compatibility: realesr-general-x4v3.pth stores weights under the
key "params" with a standard SRVGGNetCompact state_dict — same weights,
loaded directly with bare torch instead of through RealESRGANer.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class SRVGGNetCompact(nn.Module):
        """
        Compact SRVGG-style super-resolution network (matches the
        architecture realesr-general-x4v3.pth was trained with).
        7 body conv layers by default (num_conv=32 for the x4v3 checkpoint),
        PixelShuffle upsampling, with a nearest-upsampled residual add —
        this residual connection is what lets the network learn only the
        high-frequency detail on top of a cheap upsample, instead of
        reconstructing the whole image from scratch.
        """

        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                     num_conv=32, upscale=4, act_type="prelu"):
            super().__init__()
            self.upscale = upscale

            def make_act():
                if act_type == "relu":
                    return nn.ReLU(inplace=True)
                if act_type == "leakyrelu":
                    return nn.LeakyReLU(negative_slope=0.1, inplace=True)
                return nn.PReLU(num_parameters=num_feat)  # default: prelu

            body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), make_act()]
            for _ in range(num_conv):
                body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), make_act()]
            body += [nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)]
            self.body = nn.ModuleList(body)
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
            return out + base


class SRGANEnhancer:
    """
    Conditional super-resolution preprocessing stage. Same public API as
    before (.enhance(), .method, .last_triggered, .last_scale_applied) —
    this is a drop-in replacement, no changes needed in pipeline.py.
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        scale: int = 4,
        min_side_trigger: int = 260,
        target_min_side: int = 480,
        n_threads: int = 4,
    ):
        self.device = device
        self.scale = scale
        self.min_side_trigger = min_side_trigger
        self.target_min_side = target_min_side
        self.method = "bicubic_fallback"
        self.last_triggered = False
        self.last_scale_applied = 1.0

        self._model = None

        if not _TORCH_AVAILABLE:
            logger.warning("torch not installed — SRGANEnhancer will fall back to bicubic + unsharp.")
            return

        weights_path = weights_path or "weights/realesr-general-x4v3.pth"
        try:
            if device == "cpu":
                torch.set_num_threads(n_threads)

            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=32, upscale=scale, act_type="prelu",
            )
            state = torch.load(weights_path, map_location=device)
            state_dict = state.get("params", state.get("params_ema", state))
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            model.to(device)

            self._model = model
            self.method = "srgan_compact"
            logger.info("SRGANEnhancer: loaded %s on %s (standalone SRVGGNetCompact)", weights_path, device)
        except Exception as e:
            logger.warning("Failed to load SR weights (%s) — using bicubic fallback.", e, exc_info=True)
            self._model = None

    def _bicubic_fallback(self, frame_bgr: np.ndarray, scale: float) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        up = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.0)
        return cv2.addWeighted(up, 1.5, blur, -0.5, 0)

    @torch.no_grad() if _TORCH_AVAILABLE else (lambda f: f)
    def _model_infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        out = self._model(tensor)
        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out = (out * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        min_side = min(h, w)

        if min_side >= self.min_side_trigger:
            self.last_triggered = False
            self.last_scale_applied = 1.0
            return frame_bgr

        self.last_triggered = True

        if self._model is not None:
            try:
                output = self._model_infer(frame_bgr)
            except Exception as e:
                logger.warning("SR inference failed (%s) — falling back to bicubic.", e, exc_info=True)
                output = self._bicubic_fallback(frame_bgr, self.scale)
        else:
            output = self._bicubic_fallback(frame_bgr, self.scale)

        oh, ow = output.shape[:2]
        out_min_side = min(oh, ow)
        if out_min_side > self.target_min_side:
            cap_scale = self.target_min_side / out_min_side
            output = cv2.resize(output, (int(ow * cap_scale), int(oh * cap_scale)), interpolation=cv2.INTER_AREA)

        self.last_scale_applied = min(output.shape[0] / h, output.shape[1] / w)
        return output
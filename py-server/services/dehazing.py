"""
Image Dehazing Service  (Improved)
------------------------------------
Path : py-server/services/dehazing.py

Removes haze / fog / smoke using an enhanced pipeline:

  Stage 1  —  Sky-Aware Dark Channel Prior (DCP)
               • Detects sky/bright regions separately to prevent over-dehazing
               • Adaptive omega based on local haze density
               • Guided-filter transmission refinement (edge-preserving)

  Stage 2  —  Post-Processing
               • White-balance correction  (fixes DCP colour cast)
               • Gamma adjustment          (restores natural brightness)
               • Unsharp masking           (recovers lost edge sharpness)
               • Contrast stretch          (per-channel percentile clip)

  Fallback — MSRCR (Multi-Scale Retinex with Colour Restoration)
               Triggered when the image is very bright overall (sky/cloudy haze)
               where DCP performs poorly.

No model weights required. Works entirely with OpenCV + NumPy.

Usage:
    from services.dehazing import DehazingService
    dehazer = DehazingService()
    result  = dehazer.dehaze(frame)   # BGR in → BGR out
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ─── Data Class ───────────────────────────────────────────────────────────────

@dataclass
class DehazingResult:
    dehazed_frame : np.ndarray   # BGR uint8 — the output clear image
    transmission  : np.ndarray   # float32 transmission map (debug)
    atm_light     : float        # estimated atmospheric light value
    method        : str          # "dcp" | "msrcr"

    def to_meta_dict(self) -> dict:
        return {
            "method":    self.method,
            "atm_light": round(float(self.atm_light), 4),
        }


# ─── Guided Filter ────────────────────────────────────────────────────────────

def _guided_filter(
    guide  : np.ndarray,
    src    : np.ndarray,
    radius : int   = 30,
    eps    : float = 1e-3,
) -> np.ndarray:
    """Edge-preserving guided filter (O(N)) — eliminates halo artefacts."""
    r = 2 * radius + 1
    def box(x): return cv2.boxFilter(x, -1, (r, r))

    N      = box(np.ones_like(guide, np.float32))
    mI     = box(guide) / N
    mp     = box(src)   / N
    mIp    = box(guide * src) / N
    covIp  = mIp - mI * mp
    varI   = box(guide * guide) / N - mI * mI
    a      = covIp / (varI + eps)
    b      = mp - a * mI
    return box(a) / N * guide + box(b) / N


# ─── MSRCR ────────────────────────────────────────────────────────────────────

def _single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.log1p(img) - np.log1p(blur + 1e-6)


def _msrcr(frame: np.ndarray) -> np.ndarray:
    """
    Multi-Scale Retinex with Colour Restoration.
    Best for bright, milky haze where DCP over-dehaze.
    """
    img = frame.astype(np.float32) + 1.0          # avoid log(0)
    sigmas = [15, 80, 250]

    # Multi-scale retinex
    msr = sum(_single_scale_retinex(img, s) for s in sigmas) / len(sigmas)

    # Colour restoration factor
    img_sum = np.sum(img, axis=2, keepdims=True) + 1e-6
    crf     = np.log1p(125.0 * img / img_sum)

    msrcr = msr * crf

    # Per-channel normalise → [0, 255]
    out = np.zeros_like(frame, np.float32)
    for c in range(3):
        ch      = msrcr[:, :, c]
        lo, hi  = np.percentile(ch, 1), np.percentile(ch, 99)
        out[:, :, c] = np.clip((ch - lo) / (hi - lo + 1e-6) * 255, 0, 255)

    return out.astype(np.uint8)


# ─── Post-Processing Helpers ──────────────────────────────────────────────────

def _white_balance(img: np.ndarray) -> np.ndarray:
    """Simple grey-world white balance — corrects DCP colour cast."""
    result = img.astype(np.float32)
    for c in range(3):
        ch_mean = result[:, :, c].mean()
        if ch_mean > 0:
            result[:, :, c] *= (result.mean() / ch_mean)
    return np.clip(result, 0, 255).astype(np.uint8)


def _gamma_correct(img: np.ndarray, gamma: float = 0.85) -> np.ndarray:
    """Gamma correction — brightens the image naturally."""
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def _unsharp_mask(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Unsharp masking — restores edge sharpness lost during dehazing."""
    blur   = cv2.GaussianBlur(img, (0, 0), 3.0)
    sharp  = cv2.addWeighted(img, 1.0 + strength, blur, -strength, 0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _contrast_stretch(img: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    """Per-channel percentile clip + stretch for maximum contrast."""
    out = np.zeros_like(img, np.float32)
    for c in range(3):
        lo, hi = np.percentile(img[:, :, c], low), np.percentile(img[:, :, c], high)
        out[:, :, c] = np.clip((img[:, :, c].astype(np.float32) - lo) / (hi - lo + 1e-6) * 255, 0, 255)
    return out.astype(np.uint8)


# ─── Dehazing Service ─────────────────────────────────────────────────────────

class DehazingService:
    """
    Production-grade image dehazing service.

    Automatically selects the best algorithm:
      • DCP + post-processing  →  for standard outdoor haze / fog / smoke
      • MSRCR                  →  for bright sky-dominant images where DCP fails

    Constructor args:
        patch_size   : dark channel patch size (default 15)
        omega        : max haze removal strength (0–1, default 0.90)
        t_min        : transmission floor (default 0.15)
        gamma        : output gamma (< 1 = brighter, default 0.85)
        sharpen      : unsharp mask strength (default 0.6)
        msrcr_thresh : mean brightness above which MSRCR is chosen (default 0.72)
    """

    def __init__(
        self,
        patch_size   : int   = 15,
        omega        : float = 0.90,
        t_min        : float = 0.15,
        gamma        : float = 0.85,
        sharpen      : float = 0.6,
        msrcr_thresh : float = 0.72,
        clear_thresh : float = 0.08
    ):
        self.patch_size   = patch_size
        self.omega        = omega
        self.t_min        = t_min
        self.gamma        = gamma
        self.sharpen      = sharpen
        self.msrcr_thresh = msrcr_thresh
        self.clear_thresh = clear_thresh
        print(f"[Dehazing] Improved DCP+MSRCR service ready")

    # ── Public API ────────────────────────────────────────────────────────────
    def _estimate_haze_density(self, dark: np.ndarray, sky_mask: np.ndarray) -> float:
        """Mean dark-channel value over non-sky pixels; near 0 = clear, higher = hazy."""
        non_sky = ~sky_mask
        if non_sky.sum() == 0:
            return 0.0
        return float(dark[non_sky].mean())
    def dehaze(self, frame: np.ndarray) -> DehazingResult:
        img = frame.astype(np.float32) / 255.0
        mean_brightness = float(img.mean())

        # Compute dark channel + sky mask up front so we can gate on haze density
        dark = self._dark_channel(img)
        sky_mask = self._detect_sky(img, dark)
        haze_density = self._estimate_haze_density(dark, sky_mask)

        # Image is already clear — skip correction (or apply a very light touch)
        if haze_density < self.clear_thresh:
            return DehazingResult(
                dehazed_frame = frame.copy(),
                transmission  = np.ones(frame.shape[:2], np.float32),
                atm_light     = mean_brightness,
                method        = "none",
            )

        # Auto-select algorithm
        if mean_brightness > self.msrcr_thresh:
            dehazed = _msrcr(frame)
            dehazed = _contrast_stretch(dehazed)
            dehazed = _unsharp_mask(dehazed, self.sharpen)
            return DehazingResult(
                dehazed_frame = dehazed,
                transmission  = np.ones(frame.shape[:2], np.float32),
                atm_light     = mean_brightness,
                method        = "msrcr",
            )

        # Standard: Sky-Aware DCP + post-processing
        # pass dark/sky_mask through so _dcp_pipeline doesn't recompute them
        return self._dcp_pipeline(frame, img, dark, sky_mask)

    # ── DCP Pipeline ──────────────────────────────────────────────────────────

    def _dcp_pipeline(self, frame: np.ndarray, img: np.ndarray, dark: np.ndarray, sky_mask: np.ndarray) -> DehazingResult:
        # 1. Dark channel
        dark = self._dark_channel(img)

        # 2. Sky mask — bright pixels where DCP should be gentler
        sky_mask = self._detect_sky(img, dark)

        # 3. Atmospheric light (using non-sky top-bright pixels)
        atm = self._atmospheric_light(img, dark, sky_mask)

        # 4. Adaptive transmission
        t_raw = self._transmission_adaptive(img, atm, sky_mask)

        # 5. Guided filter refinement
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t_ref = _guided_filter(gray, t_raw, radius=30, eps=8e-4)
        t_ref = np.clip(t_ref, self.t_min, 1.0)

        # Soften transmission in sky regions (prevent over-dehazing the sky)
        t_ref[sky_mask] = np.maximum(t_ref[sky_mask], 0.6)

        # 6. Recover scene radiance
        dehazed = self._recover(img, t_ref, atm)
        out     = np.clip(dehazed * 255.0, 0, 255).astype(np.uint8)

        # 7. Post-processing
        out = _white_balance(out)
        out = _gamma_correct(out, self.gamma)
        out = _contrast_stretch(out, low=0.5, high=99.5)
        out = _unsharp_mask(out, self.sharpen)

        return DehazingResult(
            dehazed_frame = out,
            transmission  = t_ref,
            atm_light     = float(np.mean(atm)),
            method        = "dcp",
        )

    def _dark_channel(self, img: np.ndarray) -> np.ndarray:
        min_channel = np.min(img, axis=2)
        kernel      = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size))
        return cv2.erode(min_channel, kernel)

    def _detect_sky(self, img: np.ndarray, dark: np.ndarray) -> np.ndarray:
        brightness = img.max(axis=2)
        saturation = (img.max(axis=2) - img.min(axis=2)) / (img.max(axis=2) + 1e-6)
        # blue channel dominance catches clear blue sky; low saturation catches hazy/grey sky
        blue_dominant = (img[:, :, 0] > img[:, :, 2]) & (img[:, :, 0] > 0.4)  # BGR: channel 0 = blue
        sky_mask = (brightness > 0.6) & ((saturation < 0.25) | blue_dominant) & (dark < 0.2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        sky_mask = cv2.morphologyEx(sky_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        return sky_mask

    def _atmospheric_light(
        self,
        img      : np.ndarray,
        dark     : np.ndarray,
        sky_mask : np.ndarray,
    ) -> np.ndarray:
        """
        Estimate atmospheric light from brightest hazy (non-sky) regions.
        Falls back to full image if no non-sky pixels are available.
        """
        h, w     = dark.shape
        n_pixels = h * w
        n_bright = max(1, n_pixels // 1000)

        flat_dark = dark.flatten()
        flat_img  = img.reshape(n_pixels, 3)
        flat_sky  = sky_mask.flatten()

        # Prefer non-sky hazy pixels for more accurate atmospheric light
        non_sky_indices = np.where(~flat_sky)[0]
        if len(non_sky_indices) > n_bright:
            top_idx = non_sky_indices[np.argsort(flat_dark[non_sky_indices])[-n_bright:]]
        else:
            top_idx = np.argsort(flat_dark)[-n_bright:]

        atm = flat_img[top_idx].max(axis=0)
        return np.clip(atm, 0.1, 1.0).astype(np.float32)

    def _transmission_adaptive(
        self,
        img      : np.ndarray,
        atm      : np.ndarray,
        sky_mask : np.ndarray,
    ) -> np.ndarray:
        """
        Compute transmission with adaptive omega — stronger removal in
        non-sky regions, gentler in sky regions.
        """
        normed   = np.clip(img / (atm + 1e-6), 0, 1)
        dark_n   = self._dark_channel(normed)

        # Use lower omega in sky areas to avoid white-wash artefacts
        omega       = np.full(dark_n.shape, self.omega, dtype=np.float32)
        omega[sky_mask] = 0.5

        return (1.0 - omega * dark_n).astype(np.float32)

    def _recover(self, img: np.ndarray, t: np.ndarray, atm: np.ndarray) -> np.ndarray:
        t3  = t[:, :, np.newaxis]
        out = (img - atm) / (t3 + 1e-6) + atm
        return np.clip(out, 0, 1)

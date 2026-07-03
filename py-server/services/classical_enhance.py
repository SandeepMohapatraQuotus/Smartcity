"""
Classical Low-Light Enhancement — Gamma Correction + CLAHE
-------------------------------------------------------------
Path : py-server/services/classical_enhance.py

Lightweight, CPU-only, model-free enhancement techniques that complement
(or stand in for) the Zero-DCE++ deep model:

  • Gamma Correction
        Non-linear brightness remapping via a 256-entry lookup table:
            output = 255 * (input / 255) ** (1 / gamma)
        gamma > 1  → brightens (lifts shadows / midtones)
        gamma < 1  → darkens
        `estimate_gamma()` picks a sensible value automatically from the
        frame's current mean luminance, so callers don't have to hand-tune
        a constant for every scene.

  • CLAHE (Contrast Limited Adaptive Histogram Equalization)
        Local, tile-based contrast boost applied to the L channel of LAB
        colour space (colour/hue in a/b channels is left untouched). The
        "contrast limited" clip prevents the noise amplification that
        plain histogram equalisation causes in near-uniform regions
        (e.g. night sky, road surface).

  • auto_enhance()
        Adaptive gamma correction first (fixes global exposure), then
        CLAHE (fixes local contrast). This is the recommended default
        for CCTV footage where lighting varies a lot between cameras.

These are pure OpenCV/NumPy — no model weights, no GPU, sub-millisecond
per frame at 1080p — so they're useful as:
    (a) a fast fallback when Zero-DCE++ weights are unavailable,
    (b) a pre-conditioning step before Zero-DCE++ / downstream detectors,
    (c) a standalone lightweight enhancement mode for constrained/edge
        deployments that can't afford a CNN pass every frame.
"""

import cv2
import numpy as np


# ─── Gamma Correction ─────────────────────────────────────────────────────────

def gamma_correction(frame: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    """
    Apply gamma correction to brighten (gamma > 1) or darken (gamma < 1) a frame.

    Uses a precomputed 256-entry lookup table (cv2.LUT) so the per-pixel cost
    is a single table lookup — effectively free compared to CNN inference.

    Args:
        frame : BGR (or grayscale) uint8 numpy array.
        gamma : > 1 brightens, < 1 darkens, 1.0 = no-op.

    Returns:
        uint8 numpy array, same shape as input.
    """
    gamma = max(0.1, float(gamma))
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(frame, table)


def estimate_gamma(
    frame: np.ndarray,
    target_mean: float = 128.0,
    gamma_min: float = 0.5,
    gamma_max: float = 3.0,
) -> float:
    """
    Estimate a gamma value that pushes the frame's mean luminance toward
    `target_mean` (0-255 scale). Dark frames → gamma > 1 (brighten).
    Overexposed frames → gamma < 1 (darken).

    Solves for gamma in:  (mean_luma / 255) ** (1/gamma) * 255  =  target_mean
        ⇒  gamma = ln(mean_luma / 255) / ln(target_mean / 255)

    A dark frame (small mean_luma) relative to target_mean yields gamma > 1
    (brighten); an overexposed frame (mean_luma > target_mean) yields
    gamma < 1 (darken).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    mean_luma = max(float(np.mean(gray)), 1.0)  # avoid log(0)

    if mean_luma >= 254.0:
        return 1.0  # already blown out / flat white — gamma won't help

    gamma = np.log(mean_luma / 255.0) / np.log(target_mean / 255.0)
    return float(np.clip(gamma, gamma_min, gamma_max))


# ─── CLAHE ────────────────────────────────────────────────────────────────────

def clahe_enhance(
    frame: np.ndarray,
    clip_limit: float = 3.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Apply CLAHE to the L (lightness) channel of LAB colour space.

    Keeps colour balance intact (a/b channels untouched) while boosting
    local contrast — much less prone to noise amplification / colour shift
    than running histogram equalisation on BGR channels directly.

    Args:
        frame          : BGR uint8 numpy array.
        clip_limit     : Contrast clipping threshold. Higher = more contrast,
                          but more noise amplification. 2.0-4.0 is a good range
                          for CCTV footage.
        tile_grid_size : Grid size for local histogram equalisation.
                          Smaller tiles = more local adaptivity but can look
                          blotchy; (8, 8) is the standard default.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# ─── Combined / Auto ───────────────────────────────────────────────────────────

def auto_enhance(
    frame: np.ndarray,
    target_mean: float = 128.0,
    clip_limit: float = 3.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> dict:
    """
    Combined classical pipeline: adaptive gamma correction → CLAHE.

    Gamma fixes global under/over-exposure first; CLAHE then recovers local
    contrast/detail (number plates, faces, license plate characters) without
    re-introducing the global exposure problem gamma just fixed.

    Returns a dict (not just the frame) so callers/logging/API responses can
    report which gamma value was actually used for this frame:
        {"frame": np.ndarray, "gamma": float, "method": "gamma+clahe"}
    """
    gamma = estimate_gamma(frame, target_mean=target_mean)
    gamma_corrected = gamma_correction(frame, gamma=gamma)
    result = clahe_enhance(
        gamma_corrected, clip_limit=clip_limit, tile_grid_size=tile_grid_size
    )
    return {"frame": result, "gamma": round(gamma, 3), "method": "gamma+clahe"}
"""
ANPR — Automatic Number Plate Recognition Service
---------------------------------------------------
Path : py-server/services/anpr.py

Two-stage pipeline:
  Stage 1 — Plate localisation:
      Attempts YOLOv8n (if a plate-detection .pt is provided), then falls
      back to an OpenCV contour + aspect-ratio heuristic that works well
      on clear images without any extra weights.

  Stage 2 — OCR:
      Uses EasyOCR (preferred — pip install easyocr).
      Falls back to pytesseract if EasyOCR is not available.

Usage:
    from services.anpr import ANPRService
    anpr   = ANPRService()
    result = anpr.read_plates(frame, frame_id="f001", camera_id="cam_01")
    for plate in result.plates:
        print(plate.cleaned_text, plate.confidence)
"""

import re
import cv2
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class PlateReading:
    bbox         : list[int]    # [x1, y1, x2, y2]  in original frame
    raw_text     : str          # OCR raw output
    cleaned_text : str          # uppercase, no spaces / special chars
    confidence   : float        # OCR confidence  0.0 – 1.0
    frame_id     : Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "bbox":         self.bbox,
            "raw_text":     self.raw_text,
            "cleaned_text": self.cleaned_text,
            "confidence":   round(self.confidence, 4),
            "frame_id":     self.frame_id,
        }


@dataclass
class ANPRResult:
    frame_id  : str
    camera_id : str
    plates    : list[PlateReading] = field(default_factory=list)
    ocr_engine: str                = "none"

    def to_dict(self) -> dict:
        return {
            "frame_id":   self.frame_id,
            "camera_id":  self.camera_id,
            "plate_count": len(self.plates),
            "ocr_engine": self.ocr_engine,
            "plates":     [p.to_dict() for p in self.plates],
        }


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _clean_plate(text: str) -> str:
    """Uppercase + remove anything that's not alphanumeric."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """
    Sharpen + threshold the plate crop for better OCR accuracy.
    """
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    _, th = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


# ─── ANPR Service ─────────────────────────────────────────────────────────────

class ANPRService:
    """
    Automatic Number Plate Recognition service.

    Constructor args:
        plate_model_path : optional path to a YOLOv8 plate-detection .pt file.
                           If None, falls back to contour-based localisation.
        min_confidence   : minimum OCR confidence to include a reading (0.0–1.0)
        languages        : EasyOCR language list, default ['en']
        device           : 'auto' | 'cpu' | 'cuda'
    """

    def __init__(
        self,
        plate_model_path : Optional[str] = None,
        min_confidence   : float         = 0.3,
        languages        : list[str]     = None,
        device           : str           = "auto",
    ):
        self.min_confidence = min_confidence
        self.languages      = languages or ["en"]
        self._yolo_detector = None
        self._ocr_engine    = None
        self._ocr_name      = "none"

        import torch
        self._device = ("cuda" if torch.cuda.is_available() else "cpu") \
                       if device == "auto" else device

        # Stage 1 — optional YOLO plate localiser
        if plate_model_path:
            self._load_yolo(plate_model_path)

        # Stage 2 — OCR
        self._load_ocr()

    # ── Public API ────────────────────────────────────────────────────────────

    def read_plates(
        self,
        frame     : np.ndarray,
        frame_id  : str = "frame_0",
        camera_id : str = "cam_0",
    ) -> ANPRResult:
        """
        Detect and read all number plates in a BGR frame.

        Steps:
          1. Localise plate regions  (YOLO or contour fallback)
          2. For each ROI: preprocess → OCR → clean text
          3. Filter by min_confidence
        """
        rois   = self._localise(frame)
        plates = []

        for bbox in rois:
            x1, y1, x2, y2 = bbox
            # Guard against out-of-bounds crops
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            roi       = frame[y1:y2, x1:x2]
            processed = _preprocess_roi(roi)
            readings  = self._ocr(processed)

            for raw_text, conf in readings:
                cleaned = _clean_plate(raw_text)
                if len(cleaned) < 4 or conf < self.min_confidence:
                    continue
                plates.append(PlateReading(
                    bbox         = [x1, y1, x2, y2],
                    raw_text     = raw_text.strip(),
                    cleaned_text = cleaned,
                    confidence   = conf,
                    frame_id     = frame_id,
                ))

        return ANPRResult(
            frame_id   = frame_id,
            camera_id  = camera_id,
            plates     = plates,
            ocr_engine = self._ocr_name,
        )

    def draw(self, frame: np.ndarray, result: ANPRResult) -> np.ndarray:
        """Draw plate bounding boxes and recognised text on the frame."""
        out = frame.copy()
        for plate in result.plates:
            x1, y1, x2, y2 = plate.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 180), 2)
            label = f"{plate.cleaned_text}  ({plate.confidence:.2f})"
            # Background for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 4, y1), (0, 255, 180), -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return out

    @property
    def ocr_engine_name(self) -> str:
        return self._ocr_name

    # ── Stage 1: Localisation ─────────────────────────────────────────────────

    def _load_yolo(self, model_path: str):
        try:
            from ultralytics import YOLO
            self._yolo_detector = YOLO(model_path)
            self._yolo_detector.to(self._device)
            print(f"[ANPR] YOLO plate detector loaded: {model_path}")
        except Exception as e:
            print(f"[ANPR] Could not load YOLO plate model: {e}  → using contour fallback")
            self._yolo_detector = None

    def _localise(self, frame: np.ndarray) -> list[list[int]]:
        """Return list of [x1,y1,x2,y2] plate candidate regions."""
        if self._yolo_detector is not None:
            return self._yolo_localise(frame)
        return self._contour_localise(frame)

    def _yolo_localise(self, frame: np.ndarray) -> list[list[int]]:
        results = self._yolo_detector.predict(frame, conf=0.3, verbose=False)
        boxes   = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
        return boxes

    def _contour_localise(self, frame: np.ndarray) -> list[list[int]]:
        """
        Heuristic plate localisation using edge detection + contour filtering.

        Looks for rectangular regions with plate-like aspect ratio (2:1 to 5:1)
        and area between 0.05% and 5% of the frame.
        """
        h, w   = frame.shape[:2]
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur   = cv2.bilateralFilter(gray, 11, 17, 17)
        edges  = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours    = sorted(contours, key=cv2.contourArea, reverse=True)[:50]

        candidates  = []
        frame_area  = h * w

        for cnt in contours:
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)

            if len(approx) < 4:
                continue

            x, y, cw, ch = cv2.boundingRect(approx)
            area          = cw * ch

            # filter by area  (0.05% – 5% of frame)
            if not (0.0005 * frame_area < area < 0.05 * frame_area):
                continue

            # filter by aspect ratio  (plate-like: width > height, ratio 1.5–6)
            if ch == 0:
                continue
            ratio = cw / ch
            if not (1.5 < ratio < 6.0):
                continue

            # small padding
            pad = 4
            candidates.append([
                max(0, x - pad), max(0, y - pad),
                min(w, x + cw + pad), min(h, y + ch + pad),
            ])

        return candidates

    # ── Stage 2: OCR ─────────────────────────────────────────────────────────

    def _load_ocr(self):
        """Try EasyOCR first, fall back to pytesseract."""
        try:
            import easyocr
            gpu = self._device != "cpu"
            self._ocr_engine = easyocr.Reader(self.languages, gpu=gpu, verbose=False)
            self._ocr_name   = "easyocr"
            print(f"[ANPR] OCR engine: EasyOCR  (gpu={gpu})")
        except ImportError:
            print("[ANPR] EasyOCR not found — trying pytesseract ...")
            try:
                import pytesseract
                self._ocr_engine = pytesseract
                self._ocr_name   = "pytesseract"
                print("[ANPR] OCR engine: pytesseract")
            except ImportError:
                print("[ANPR] WARNING: No OCR engine found. "
                      "Install easyocr (pip install easyocr) for best results.")
                self._ocr_name = "none"

    def _ocr(self, processed: np.ndarray) -> list[tuple[str, float]]:
        """
        Run OCR on a preprocessed (grayscale) plate ROI.
        Returns list of (text, confidence) tuples.
        """
        if self._ocr_name == "easyocr":
            return self._easyocr_read(processed)
        if self._ocr_name == "pytesseract":
            return self._tesseract_read(processed)
        return []

    def _easyocr_read(self, img: np.ndarray) -> list[tuple[str, float]]:
        try:
            results = self._ocr_engine.readtext(img, detail=1, paragraph=False)
            return [(text, conf) for (_, text, conf) in results]
        except Exception as e:
            print(f"[ANPR] EasyOCR error: {e}")
            return []

    def _tesseract_read(self, img: np.ndarray) -> list[tuple[str, float]]:
        try:
            import pytesseract
            config = "--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            text   = pytesseract.image_to_string(img, config=config).strip()
            return [(text, 0.7)] if text else []
        except Exception as e:
            print(f"[ANPR] Tesseract error: {e}")
            return []

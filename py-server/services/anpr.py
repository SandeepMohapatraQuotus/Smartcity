

"""
ANPR — Automatic Number Plate Recognition Service
---------------------------------------------------
Path : py-server/services/anpr.py

Two-stage pipeline:
  Stage 1 — Plate localisation:
      Uses YOLOv8 (if a plate-detection .pt is provided via plate_model_path —
      RECOMMENDED, much more robust on real street/night footage), then
      falls back to an OpenCV contour + aspect-ratio heuristic that only
      works reliably on clean, high-contrast, head-on plate photos.

  Stage 2 — OCR:
      Uses EasyOCR (preferred — pip install easyocr).
      Falls back to pytesseract if EasyOCR is not available.

Usage:
    from services.anpr import ANPRService
    anpr   = ANPRService(plate_model_path="weights/license_plate_yolov8n.pt")
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
    Upscale + sharpen the plate crop for better OCR accuracy.

    MODIFIED: previously applied an OTSU binary threshold after sharpening.
    EasyOCR is a neural model trained on natural scene text and generally
    performs WORSE on hard-binarized crops than on plain grayscale —
    especially under uneven lighting, glare, or motion blur (e.g. night
    traffic footage), where OTSU can pick a bad threshold and wipe out
    legible characters. Binarization was silently causing empty OCR reads
    even when plates were clearly visible. Returning grayscale instead.
    """
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray  = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    return sharp


def _deduplicate_plates(plates: list[PlateReading], iou_thresh: float = 0.3) -> list[PlateReading]:
    """Remove overlapping duplicate detections, keeping the highest-confidence one."""
    if not plates:
        return plates
    plates = sorted(plates, key=lambda p: p.confidence, reverse=True)
    keep = []
    for p in plates:
        is_dup = False
        for k in keep:
            if _iou(p.bbox, k.bbox) > iou_thresh:
                is_dup = True
                break
        if not is_dup:
            keep.append(p)
    return keep


def _iou(box_a: list[int], box_b: list[int]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


# ─── ANPR Service ─────────────────────────────────────────────────────────────

class ANPRService:
    """
    Automatic Number Plate Recognition service.

    Constructor args:
        plate_model_path : path to a YOLOv8 plate-detection .pt file.
                           STRONGLY RECOMMENDED for real footage — the contour
                           fallback only works on clean, high-contrast, head-on
                           plate shots. If None, falls back to contour heuristic.
        min_confidence   : minimum OCR confidence to include a reading (0.0–1.0)
        languages        : EasyOCR language list, default ['en']
        device           : 'auto' | 'cpu' | 'cuda'
    """

    def __init__(
        self,
        plate_model_path : Optional[str] = None,
        min_confidence   : float         = 0.10,
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

    def read_plates(self, frame, frame_id="frame_0", camera_id="cam_0") -> ANPRResult:
        rois = self._localise(frame)
        plates = []

        for bbox in rois:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue

            roi = frame[y1:y2, x1:x2]
            processed = _preprocess_roi(roi)
            readings = self._ocr(processed)

            if not readings:
                continue

            combined_text = " ".join(text for text, conf in readings)
            avg_conf = sum(conf for _, conf in readings) / len(readings)

            cleaned = _clean_plate(combined_text)
            if len(cleaned) < 4 or avg_conf < self.min_confidence:
                continue

            plates.append(PlateReading(
                bbox=[x1, y1, x2, y2],
                raw_text=combined_text.strip(),
                cleaned_text=cleaned,
                confidence=avg_conf,
                frame_id=frame_id,
            ))

        plates = _deduplicate_plates(plates)
        return ANPRResult(frame_id=frame_id, camera_id=camera_id, plates=plates, ocr_engine=self._ocr_name)

    def read_plates_in_vehicles(
        self,
        frame              : np.ndarray,
        vehicle_boxes      : list,   # list of [x1, y1, x2, y2] from VehicleDetector
        frame_id           : str = "frame_0",
        camera_id          : str = "cam_0",
    ) -> ANPRResult:
        """
        Run plate localisation + OCR only within given vehicle bounding boxes,
        instead of scanning the entire frame. Dramatically reduces false
        positives from background clutter (signs, light streaks, trim, etc.)
        and is the primary path used by the live pipeline.

        MODIFIED: now calls self._localise() (YOLO-or-contour, whichever is
        configured) instead of hardcoding self._contour_localise(). Previously
        this bypassed the YOLO plate detector entirely even when one was loaded.
        """
        plates = []

        for vbox in vehicle_boxes:
            vx1, vy1, vx2, vy2 = vbox
            vx1, vy1 = max(0, vx1), max(0, vy1)
            vx2, vy2 = min(frame.shape[1], vx2), min(frame.shape[0], vy2)
            if vx2 <= vx1 or vy2 <= vy1:
                continue

            # Plates are almost always in the lower half of a vehicle's bbox —
            # narrowing further improves localisation precision either way.
            veh_h = vy2 - vy1
            crop_y1 = vy1 + int(veh_h * 0.4)
            vehicle_crop = frame[crop_y1:vy2, vx1:vx2]
            if vehicle_crop.size == 0:
                continue

            # Run configured localiser (YOLO if available, else contour), but
            # only within this small crop
            local_rois = self._localise(vehicle_crop)

            for lx1, ly1, lx2, ly2 in local_rois:
                # Translate crop-local coords back to full-frame coords
                fx1, fy1 = vx1 + lx1, crop_y1 + ly1
                fx2, fy2 = vx1 + lx2, crop_y1 + ly2

                roi = frame[fy1:fy2, fx1:fx2]
                if roi.size == 0:
                    continue
                processed = _preprocess_roi(roi)
                readings = self._ocr(processed)
                if not readings:
                    continue

                combined_text = " ".join(text for text, conf in readings)
                avg_conf = sum(conf for _, conf in readings) / len(readings)
                cleaned = _clean_plate(combined_text)
                if len(cleaned) < 4 or avg_conf < self.min_confidence:
                    continue

                plates.append(PlateReading(
                    bbox=[fx1, fy1, fx2, fy2],
                    raw_text=combined_text.strip(),
                    cleaned_text=cleaned,
                    confidence=avg_conf,
                    frame_id=frame_id,
                ))

        plates = _deduplicate_plates(plates)
        return ANPRResult(frame_id=frame_id, camera_id=camera_id, plates=plates, ocr_engine=self._ocr_name)

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
        """
        MODIFIED: PyTorch 2.6+ defaults torch.load(weights_only=True), which
        blocks unpickling many classes referenced by community-trained YOLO
        checkpoints (ultralytics internals, torch.nn.Sequential, etc.).
        Allowlisting classes one at a time is impractical — checkpoints can
        reference dozens of them. Since model_path is a path YOU control
        (only ever point this at weights from a source you trust), we
        temporarily force weights_only=False for just this one load call,
        then immediately restore the original torch.load behavior so nothing
        else in the app is affected.
        """
        import torch
        _orig_torch_load = torch.load

        def _trusted_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_torch_load(*args, **kwargs)

        try:
            from ultralytics import YOLO
            torch.load = _trusted_load
            self._yolo_detector = YOLO(model_path)
            self._yolo_detector.to(self._device)
            print(f"[ANPR] YOLO plate detector loaded: {model_path}")
        except Exception as e:
            print(f"[ANPR] Could not load YOLO plate model: {e}  → using contour fallback")
            self._yolo_detector = None
        finally:
            torch.load = _orig_torch_load  # always restore, even on failure

    def _localise(self, frame: np.ndarray) -> list[list[int]]:
        """Return list of [x1,y1,x2,y2] plate candidate regions."""
        if self._yolo_detector is not None:
            return self._yolo_localise(frame)
        return self._contour_localise(frame)

    def _yolo_localise(self, frame: np.ndarray) -> list[list[int]]:
        """
        MODIFIED: conf lowered from 0.3 → 0.15. On blurry / low-light / angled
        footage the model's confidence naturally sits lower even on real
        plates; 0.3 was silently dropping true positives. Downstream OCR
        min_confidence + _deduplicate_plates() still filter out junk, so this
        is safe to loosen at the detection stage.
        """
        if frame.size == 0:
            return []
        results = self._yolo_detector.predict(frame, conf=0.15, verbose=False)
        boxes   = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                boxes.append([x1, y1, x2, y2])
        return boxes

    def _contour_localise(self, frame: np.ndarray) -> list[list[int]]:
        """
        Fallback heuristic localiser — used only when no YOLO plate model is
        configured. Relies on Canny edges + boundingRect instead of the
        stricter approxPolyDP quad-fitting, which was rejecting real plates
        under blur/glare/angle. Only reliable on clean, high-contrast,
        head-on shots — prefer the YOLO path for real street/night footage.
        """
        if frame.size == 0:
            return []
        h, w   = frame.shape[:2]
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur   = cv2.bilateralFilter(gray, 11, 17, 17)
        edges  = cv2.Canny(blur, 30, 200)

        # RETR_LIST catches contours that RETR_TREE's nesting logic can miss
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours    = sorted(contours, key=cv2.contourArea, reverse=True)[:100]  # widened from 50

        candidates  = []
        frame_area  = h * w

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)  # skip approxPolyDP's strict 4-corner requirement
            area = cw * ch

            # filter by area (0.03% – 40% of frame) — widened lower bound
            if not (0.0003 * frame_area < area < 0.40 * frame_area):
                continue

            if ch == 0:
                continue
            ratio = cw / ch
            if not (1.2 < ratio < 7.0):  # widened from 1.5–6.0, plates get sheared at angle
                continue

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
"""
Person Detector — YOLOv8
--------------------------
Path   : clashifiers/person_detector/main.py
Model  : YOLOv8m  (COCO pretrained — class 0 = person, no custom training needed)
Input  : BGR frame  (numpy array from OpenCV)
Output : PersonDetectionResult  →  list of PersonDetection per frame

COCO person class used:
    0 → person

Enhancements applied:
    - conf_threshold lowered 0.4 → 0.30 for better recall on occluded/distant people
    - iou_threshold raised 0.45 → 0.50 for tighter NMS in dense crowd scenes
    - BotSORT tracker (vs ByteTrack) for robust ID persistence through occlusion
    - Minimum bbox height filter to kill shadow/bag false positives
    - adaptive_detect() method: scales input resolution based on crowd density
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional



@dataclass
class PersonDetection:
    bbox       : list[int]         # [x1, y1, x2, y2]
    confidence : float
    track_id   : Optional[int] = None
    frame_id   : Optional[str] = None

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> dict:
        return {
            "bbox":       self.bbox,
            "confidence": round(self.confidence, 4),
            "track_id":   self.track_id,
            "frame_id":   self.frame_id,
            "center":     list(self.center),
            "area":       self.area,
            "width":      self.width,
            "height":     self.height,
        }


@dataclass
class PersonDetectionResult:
    frame_id     : str
    camera_id    : str
    detections   : list[PersonDetection] = field(default_factory=list)
    person_count : int                   = 0

    def to_dict(self) -> dict:
        return {
            "frame_id":     self.frame_id,
            "camera_id":    self.camera_id,
            "person_count": self.person_count,
            "detections":   [d.to_dict() for d in self.detections],
        }


# ─── Detector ─────────────────────────────────────────────────────────────────

class PersonDetector:
    """
    YOLOv8-based person detector with optional ByteTrack tracking.

    Reuses the same YOLOv8m.pt already present — COCO class 0 = person.
    No extra model weights needed.

    Model size options:
        yolov8n  fastest  (edge / low resource)
        yolov8s
        yolov8m  ← default, best balance
        yolov8l
        yolov8x  highest accuracy

    Usage:
        from clashifiers.person_detector.main import PersonDetector
        detector = PersonDetector()
        result   = detector.detect(frame, frame_id="f001", camera_id="cam_01")
    """

    PERSON_CLASS_ID = 0
    BOX_COLOR       = (0, 165, 255)   # orange-ish

    # Minimum bounding-box height (pixels) for a detection to be kept.
    # Eliminates bags, shadows, and partial-body blobs that YOLO hallucinates
    # at low confidence thresholds. Tune per camera resolution.
    MIN_PERSON_HEIGHT = 40

    def __init__(
        self,
        model_size      : str   = "yolov8m",
        conf_threshold  : float = 0.30,   # ↓ from 0.4 — better recall on occluded/distant people
        iou_threshold   : float = 0.50,   # ↑ from 0.45 — tighter NMS for dense crowd scenes
        enable_tracking : bool  = True,
        device          : str   = "auto",
        tracker         : str   = "botsort.yaml",  # BotSORT > ByteTrack on occlusion
        min_person_height: int  = 40,   # px — filter out tiny false-positive blobs
    ):
        self.conf_threshold   = conf_threshold
        self.iou_threshold    = iou_threshold
        self.enable_tracking  = enable_tracking
        self.tracker          = tracker
        self.MIN_PERSON_HEIGHT = min_person_height

        import torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
                      if device == "auto" else device

        self._load_model(model_size)

    def _load_model(self, model_size: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(f"{model_size}.pt")
            self.model.to(self.device)
            print(f"[PersonDetector] {model_size}.pt loaded on {self.device}")
        except ImportError:
            raise ImportError("Run: pip install ultralytics")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        frame     : np.ndarray,
        frame_id  : str = "frame_0",
        camera_id : str = "cam_0",
    ) -> PersonDetectionResult:
        """Run person detection (+ BotSORT tracking) on a single BGR frame."""
        if self.enable_tracking:
            raw = self.model.track(
                frame,
                classes = [self.PERSON_CLASS_ID],
                conf    = self.conf_threshold,
                iou     = self.iou_threshold,
                persist = True,
                tracker = self.tracker,   # BotSORT for better occlusion handling
                verbose = False,
            )
        else:
            raw = self.model.predict(
                frame,
                classes = [self.PERSON_CLASS_ID],
                conf    = self.conf_threshold,
                iou     = self.iou_threshold,
                verbose = False,
            )
        return self._parse(raw[0], frame_id, camera_id)

    def adaptive_detect(
        self,
        frame     : np.ndarray,
        frame_id  : str = "frame_0",
        camera_id : str = "cam_0",
        person_count_hint: int = 0,
    ) -> "PersonDetectionResult":
        """
        Scale the input frame resolution based on crowd density before detection.

        Heuristic (mirrors face recogniser adaptive_detect logic):
          0-3  people hint → native resolution  (fast path)
          4-8  people hint → upscale to 1280px wide (balanced)
          9+   people hint → upscale to 1920px wide (dense crowd / wide CCTV shot)

        Upscaling helps detect small/distant people that YOLO misses at native
        resolution when they occupy only a few dozen pixels in the frame.
        """
        h, w = frame.shape[:2]

        if person_count_hint >= 9:
            target_w = 1920
        elif person_count_hint >= 4:
            target_w = 1280
        else:
            # No upscaling needed — run at native resolution
            return self.detect(frame, frame_id, camera_id)

        if w < target_w:  # only upscale, never downscale here
            scale   = target_w / w
            new_w   = target_w
            new_h   = int(round(h * scale))
            frame   = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        result = self.detect(frame, frame_id, camera_id)

        # Re-scale bboxes back to original frame coordinates
        if w < target_w:
            scale_back = w / target_w
            for det in result.detections:
                det.bbox = [
                    int(det.bbox[0] * scale_back),
                    int(det.bbox[1] * scale_back),
                    int(det.bbox[2] * scale_back),
                    int(det.bbox[3] * scale_back),
                ]
        return result

    def draw(self, frame: np.ndarray, result: PersonDetectionResult) -> np.ndarray:
        """Return annotated BGR frame with bounding boxes and person count."""
        out = frame.copy()
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), self.BOX_COLOR, 2)
            tag = f"[{det.track_id}] person {det.confidence:.2f}" \
                  if det.track_id is not None else f"person {det.confidence:.2f}"
            cv2.putText(out, tag, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.BOX_COLOR, 2)

        cv2.putText(out, f"Persons: {result.person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return out

    # ── Internal ──────────────────────────────────────────────────────────────

    def _parse(self, raw, frame_id: str, camera_id: str) -> PersonDetectionResult:
        detections = []

        if raw.boxes is None:
            return PersonDetectionResult(frame_id, camera_id, detections, 0)

        track_ids = raw.boxes.id.int().tolist() \
                    if raw.boxes.id is not None else [None] * len(raw.boxes)

        for box, tid in zip(raw.boxes, track_ids):
            cid = int(box.cls[0])
            if cid != self.PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # ── Min-size filter: skip tiny blobs (shadows, bags, distant noise) ──
            if (y2 - y1) < self.MIN_PERSON_HEIGHT:
                continue

            detections.append(PersonDetection(
                bbox       = [x1, y1, x2, y2],
                confidence = float(box.conf[0]),
                track_id   = tid,
                frame_id   = frame_id,
            ))

        return PersonDetectionResult(
            frame_id     = frame_id,
            camera_id    = camera_id,
            detections   = detections,
            person_count = len(detections),
        )

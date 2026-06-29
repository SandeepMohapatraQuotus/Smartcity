"""
Vehicle Detector — YOLOv8
--------------------------
Path   : classifiers/vechile_detector/main.py
Model  : YOLOv8m  (COCO pretrained — no custom training needed)
Input  : BGR frame  (numpy array from OpenCV)
Output : VehicleDetectionResult  →  list of VehicleDetection per frame

COCO vehicle classes used:
    2 → car  |  3 → motorcycle  |  5 → bus  |  7 → truck
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class VehicleDetection:
    bbox       : list[int]         # [x1, y1, x2, y2]
    label      : str               # car / truck / bus / motorcycle / bicycle
    confidence : float
    class_id   : int
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

    def to_dict(self) -> dict:
        return {
            "bbox":       self.bbox,
            "label":      self.label,
            "confidence": round(self.confidence, 4),
            "class_id":   self.class_id,
            "track_id":   self.track_id,
            "center":     list(self.center),
            "area":       self.area,
        }


@dataclass
class VehicleDetectionResult:
    frame_id      : str
    camera_id     : str
    detections    : list[VehicleDetection] = field(default_factory=list)
    vehicle_count : dict                   = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "frame_id":      self.frame_id,
            "camera_id":     self.camera_id,
            "total":         len(self.detections),
            "vehicle_count": self.vehicle_count,
            "detections":    [d.to_dict() for d in self.detections],
        }


# ─── Detector ─────────────────────────────────────────────────────────────────

class VehicleDetector:
    """
    YOLOv8-based vehicle detector with optional ByteTrack tracking.

    Model size options:
        yolov8n  ~200 FPS  (edge / low resource)
        yolov8s  ~150 FPS
        yolov8m  ~80  FPS  ← default, best balance
        yolov8l  ~50  FPS
        yolov8x  ~30  FPS  (highest accuracy)

    Usage:
        from classifiers.vechile_detector.main import VehicleDetector
        detector = VehicleDetector()
        result   = detector.detect(frame, frame_id="f001", camera_id="cam_01")
    """

    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    BICYCLE_CLASS   = {1: "bicycle"}

    COLORS = {
        "car":        (0, 255, 0),
        "truck":      (255, 140, 0),
        "bus":        (0, 0, 255),
        "motorcycle": (255, 0, 255),
        "bicycle":    (0, 255, 255),
    }

    def __init__(
        self,
        model_size      : str   = "yolov8m",
        conf_threshold  : float = 0.4,
        iou_threshold   : float = 0.45,
        include_bicycle : bool  = False,
        enable_tracking : bool  = True,
        device          : str   = "auto",
    ):
        self.conf_threshold  = conf_threshold
        self.iou_threshold   = iou_threshold
        self.enable_tracking = enable_tracking

        self.target_classes = dict(self.VEHICLE_CLASSES)
        if include_bicycle:
            self.target_classes.update(self.BICYCLE_CLASS)
        self.class_ids = list(self.target_classes.keys())

        import torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
                      if device == "auto" else device

        self._load_model(model_size)

    def _load_model(self, model_size: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(f"{model_size}.pt")
            self.model.to(self.device)
            print(f"[VehicleDetector] {model_size}.pt loaded on {self.device}")
        except ImportError:
            raise ImportError("Run: pip install ultralytics")

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        frame     : np.ndarray,
        frame_id  : str = "frame_0",
        camera_id : str = "cam_0",
    ) -> VehicleDetectionResult:
        """Run detection (+ tracking) on a single BGR frame."""
        if self.enable_tracking:
            raw = self.model.track(
                frame,
                classes = self.class_ids,
                conf    = self.conf_threshold,
                iou     = self.iou_threshold,
                persist = True,
                verbose = False,
            )
        else:
            raw = self.model.predict(
                frame,
                classes = self.class_ids,
                conf    = self.conf_threshold,
                iou     = self.iou_threshold,
                verbose = False,
            )
        return self._parse(raw[0], frame_id, camera_id)

    def draw(self, frame: np.ndarray, result: VehicleDetectionResult) -> np.ndarray:
        """Return annotated BGR frame with bounding boxes and labels."""
        out = frame.copy()
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = self.COLORS.get(det.label, (200, 200, 200))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            tag = f"[{det.track_id}] {det.label} {det.confidence:.2f}" \
                  if det.track_id is not None else f"{det.label} {det.confidence:.2f}"
            cv2.putText(out, tag, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        y = 30
        for label, count in result.vehicle_count.items():
            cv2.putText(out, f"{label}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y += 25
        return out

    # ── Internal ──────────────────────────────────────────────────────────────

    def _parse(self, raw, frame_id: str, camera_id: str) -> VehicleDetectionResult:
        detections    = []
        vehicle_count = {v: 0 for v in self.target_classes.values()}

        if raw.boxes is None:
            return VehicleDetectionResult(frame_id, camera_id, detections, vehicle_count)

        track_ids = raw.boxes.id.int().tolist() \
                    if raw.boxes.id is not None else [None] * len(raw.boxes)

        for box, tid in zip(raw.boxes, track_ids):
            cid   = int(box.cls[0])
            label = self.target_classes.get(cid)
            if not label:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append(VehicleDetection(
                bbox       = [x1, y1, x2, y2],
                label      = label,
                confidence = float(box.conf[0]),
                class_id   = cid,
                track_id   = tid,
                frame_id   = frame_id,
            ))
            vehicle_count[label] = vehicle_count.get(label, 0) + 1

        return VehicleDetectionResult(frame_id, camera_id, detections, vehicle_count)


# ─── Fine-Tune Helper ─────────────────────────────────────────────────────────

def fine_tune(
    data_yaml  : str,
    base_model : str = "yolov8m.pt",
    epochs     : int = 50,
    img_size   : int = 640,
    output_dir : str = "runs/vehicle_finetune",
):
    """
    Fine-tune YOLOv8 on your own traffic footage (YOLO format).

    dataset.yaml example:
        path: ./data
        train: images/train
        val:   images/val
        names: {0: car, 1: motorcycle, 2: bus, 3: truck}

    Usage:
        from classifiers.vechile_detector.main import fine_tune
        fine_tune("data/dataset.yaml", epochs=50)
    """
    from ultralytics import YOLO
    YOLO(base_model).train(
        data=data_yaml, epochs=epochs, imgsz=img_size,
        project=output_dir, name="exp",
    )
    print(f"[VehicleDetector] Weights → {output_dir}/exp/weights/best.pt")
"""
Smart City Pipeline
--------------------
Path : py-server/pipeline.py

Orchestrates all three classifiers in order:

  BGR Frame
      │
      ▼
  DayNightClassifier  ──night──▶  enhance_night_frame()
      │ day / enhanced                    │
      └──────────────────────────────────┘
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
     VehicleDetector        FaceRecogniser
             │                     │
             └──────────┬──────────┘
                        ▼
                  FrameEvent  (pushed to Kafka / DB / API response)
"""

import cv2
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from clashifiers.day_night.main         import DayNightClassifier
from clashifiers.vechile_detector.main   import VehicleDetector, VehicleDetectionResult
from clashifiers.face_recognization.main import FaceRecogniser, Watchlist, FaceResult
from clashifiers.person_detector.main   import PersonDetector, PersonDetectionResult
from services.zero_dce  import ZeroDCEEnhancer
from services.anpr      import ANPRService, ANPRResult
from services.dehazing  import DehazingService


# ─── Frame Event ──────────────────────────────────────────────────────────────

@dataclass
class FrameEvent:
    """
    Single unified event record per frame.
    Written to Analytics DB / Kafka topic / API response.
    """
    frame_id  : str
    camera_id : str
    timestamp : float
    day_night : dict
    enhanced  : bool                             = False
    vehicles  : Optional[VehicleDetectionResult] = None
    persons   : Optional[PersonDetectionResult]  = None
    plates    : Optional[ANPRResult]             = None
    faces     : list[FaceResult]                 = field(default_factory=list)
    alerts    : list[dict]                       = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame_id":  self.frame_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "day_night": self.day_night,
            "enhanced":  self.enhanced,
            "vehicles":  self.vehicles.to_dict() if self.vehicles else None,
            "persons":   self.persons.to_dict()  if self.persons  else None,
            "plates":    self.plates.to_dict()   if self.plates   else None,
            "faces":     [f.to_dict() for f in self.faces],
            "alerts":    self.alerts,
        }


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class SmartCityPipeline:
    """
    Main pipeline class — instantiated once at FastAPI startup.

    Usage:
        pipeline = SmartCityPipeline()
        event    = pipeline.process_frame(frame)
        print(event.to_dict())
    """

    def __init__(
        self,
        camera_id           : str   = "cam_01",
        vehicle_model_size  : str   = "yolov8m",
        face_ctx_id         : int   = -1,        # -1 = CPU, 0 = GPU
        day_night_weights   : Optional[str] = None,
        watchlist_path      : Optional[str] = None,
        face_sim_threshold  : float = 0.55,
    ):
        self.camera_id = camera_id
        self._frame_idx = 0

        print("[Pipeline] Loading Day/Night Classifier ...")
        self.day_night = DayNightClassifier(model_path=day_night_weights)

        print("[Pipeline] Loading Vehicle Detector (YOLOv8) ...")
        self.vehicles = VehicleDetector(
            model_size      = vehicle_model_size,
            enable_tracking = True,
        )

        print("[Pipeline] Loading Face Recogniser (InsightFace) ...")
        self.recogniser = FaceRecogniser(ctx_id=face_ctx_id)
        self.watchlist  = Watchlist(similarity_threshold=face_sim_threshold)

        if watchlist_path:
            import os
            if os.path.exists(watchlist_path):
                self.watchlist.load(watchlist_path)

        print("[Pipeline] Loading Night Enhancer (Zero-DCE++) ...")
        self.enhancer = ZeroDCEEnhancer()

        print("[Pipeline] Loading Person Detector (YOLOv8) ...")
        self.person_detector = PersonDetector(
            model_size      = vehicle_model_size,
            enable_tracking = True,
        )

        print("[Pipeline] Loading ANPR Service (EasyOCR) ...")
        self.anpr = ANPRService()

        print("[Pipeline] Loading Dehazing Service (Dark Channel Prior) ...")
        self.dehazer = DehazingService()

        print("[Pipeline] All models ready.\n")

    # ── Single Frame ──────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> FrameEvent:
        """
        Run full pipeline on one BGR frame.

        Steps:
          1. Day/Night classification
          2. Night enhancement  (if needed)
          3. Vehicle detection  (YOLOv8 + ByteTrack)
          4. Face recognition   (RetinaFace + ArcFace + watchlist)
          5. Alert collection

        Returns FrameEvent with all results merged.
        """
        self._frame_idx += 1
        frame_id  = f"frame_{self._frame_idx:06d}"
        timestamp = time.time()

        # 1. Day / Night
        dn       = self.day_night.predict(frame)
        working  = frame
        enhanced = False

        # 2. Enhance if night  (Zero-DCE++ or CLAHE fallback)
        if dn["route_to_enhancement"]:
            working  = self.enhancer.enhance(frame)
            enhanced = True

        # 3. Vehicle detection
        vehicle_result = self.vehicles.detect(
            working,
            frame_id  = frame_id,
            camera_id = self.camera_id,
        )

        # 4. Person detection
        person_result = self.person_detector.detect(
            working,
            frame_id  = frame_id,
            camera_id = self.camera_id,
        )

        # 5. ANPR — number plate reading
        anpr_result = self.anpr.read_plates(
            working,
            frame_id  = frame_id,
            camera_id = self.camera_id,
        )

        # 6. Face recognition + watchlist
        face_results = self.recogniser.recognise(
            working,
            self.watchlist,
            frame_id  = frame_id,
            camera_id = self.camera_id,
        )

        # 7. Alerts
        alerts = [
            {
                "type":       "face_watchlist_hit",
                "person_id":  fr.match.person_id,
                "name":       fr.match.name,
                "similarity": fr.match.similarity,
                "frame_id":   frame_id,
                "camera_id":  self.camera_id,
                "timestamp":  timestamp,
            }
            for fr in face_results
            if fr.match and fr.match.is_match
        ]

        return FrameEvent(
            frame_id  = frame_id,
            camera_id = self.camera_id,
            timestamp = timestamp,
            day_night = dn,
            enhanced  = enhanced,
            vehicles  = vehicle_result,
            persons   = person_result,
            plates    = anpr_result,
            faces     = face_results,
            alerts    = alerts,
        )

    def annotate(self, frame: np.ndarray, event: FrameEvent) -> np.ndarray:
        """Draw all detection overlays on the frame — for dashboard preview."""
        out = frame.copy()
        if event.vehicles:
            out = self.vehicles.draw(out, event.vehicles)
        if event.faces:
            out = self.recogniser.draw(out, event.faces)

        label = f"{event.day_night['label'].upper()}  {event.day_night['confidence']:.2f}"
        if event.enhanced:
            label += "  [ENHANCED]"
        cv2.putText(out, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return out

    # ── Live Stream ───────────────────────────────────────────────────────────

    def run_stream(
        self,
        source      : str | int = 0,
        show_window : bool      = True,
        on_event                = None,   # callback(FrameEvent) → Kafka / DB
    ):
        """
        Process a live RTSP stream or webcam.

        Args:
            source      : RTSP URL  e.g. "rtsp://admin:pass@192.168.1.10:554/stream1"
                          or webcam index 0
            show_window : Show annotated preview (disable on headless servers)
            on_event    : Optional callback — plug your Kafka producer here
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {source}")

        print(f"[Pipeline] Stream started: {source}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[Pipeline] Stream ended / frame lost.")
                    break

                event = self.process_frame(frame)

                if on_event:
                    on_event(event)

                if show_window:
                    cv2.imshow("Smart City Pipeline", self.annotate(frame, event))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[Pipeline] Stream closed.")
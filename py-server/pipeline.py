

import cv2
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
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
        # ── Performance knobs ──────────────────────────────────────────────────
        anpr_interval       : int   = 5,     # run ANPR every N frames (0 = every frame)
        inference_max_side  : int   = 960,   # resize longer edge to this before inference
                                             # (0 = disabled, use original resolution)
        face_det_size       : int   = 320,   # InsightFace detection grid (320 or 640)
        n_threads           : int   = 4,     # parallel inference workers
        # ── ANPR knobs  ← NEW ────────────────────────────────────────────────
        plate_model_path    : Optional[str] = None,   # path to YOLOv8 plate-detection .pt
                                                        # (None → falls back to contour heuristic)
        anpr_min_confidence : float = 0.10,
    ):
        self.camera_id = camera_id
        self._frame_idx = 0

        # ANPR throttle state
        self._anpr_interval    = anpr_interval
        self._last_anpr_result : Optional[ANPRResult] = None

        # Frame downscale cap (0 = off)
        self._inference_max_side = inference_max_side

        print("[Pipeline] Loading Day/Night Classifier ...")
        self.day_night = DayNightClassifier(model_path=day_night_weights)

        print("[Pipeline] Loading Vehicle Detector (YOLOv8) ...")
        self.vehicles = VehicleDetector(
            model_size      = vehicle_model_size,
            enable_tracking = True,
        )

        print("[Pipeline] Loading Face Recogniser (InsightFace) ...")
        self.recogniser = FaceRecogniser(
            ctx_id   = face_ctx_id,
            det_size = (face_det_size, face_det_size),   # smaller → faster
        )
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
        self.anpr = ANPRService(
            plate_model_path = plate_model_path,   # NEW — enables YOLO plate localiser
            min_confidence   = anpr_min_confidence,
        )

        print("[Pipeline] Loading Dehazing Service (Dark Channel Prior) ...")
        self.dehazer = DehazingService()

        # Persistent thread pool — created once, shared across all frames
        self._pool = ThreadPoolExecutor(max_workers=n_threads,
                                        thread_name_prefix="sc_infer")

        print(
            f"[Pipeline] All models ready.\n"
            f"           anpr_interval={anpr_interval}  "
            f"inference_max_side={inference_max_side}  "
            f"face_det_size={face_det_size}  "
            f"n_threads={n_threads}  "
            f"plate_model={'YOLO (' + plate_model_path + ')' if plate_model_path else 'contour fallback'}\n"
        )

    # ── Frame helpers ─────────────────────────────────────────────────────────

    def _maybe_downscale(self, frame: np.ndarray) -> np.ndarray:
        """
        Resize frame so its longer side ≤ inference_max_side.
        Returns original frame unchanged if downscaling is disabled or not needed.
        """
        if self._inference_max_side <= 0:
            return frame
        h, w = frame.shape[:2]
        longer = max(h, w)
        if longer <= self._inference_max_side:
            return frame
        scale  = self._inference_max_side / longer
        new_w  = int(round(w * scale))
        new_h  = int(round(h * scale))
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # ── Single Frame ──────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> FrameEvent:
        """
        Run full pipeline on one BGR frame with parallel inference.

        Steps:
          1. Day/Night classification          (cheap heuristic first)
          2. Night enhancement                 (only when needed)
          3. Optional frame downscale          (speeds up all downstream models)
          4. Parallel inference:
               ├─ Vehicle detection  (YOLOv8 + ByteTrack)
               ├─ Person detection   (YOLOv8 + ByteTrack)
               └─ Face recognition   (RetinaFace + ArcFace + watchlist)
          4b. ANPR (sequential, AFTER vehicle detection)
               Runs only inside each detected vehicle's bounding box —
               see services/anpr.py:read_plates_in_vehicles(). This needs
               vehicle_result, so it can no longer run in the same parallel
               batch as vehicle detection itself.
          5. Alert collection

        Returns FrameEvent with all results merged.
        """
        self._frame_idx += 1
        frame_id  = f"frame_{self._frame_idx:06d}"
        timestamp = time.time()

        # ── 1. Day / Night ────────────────────────────────────────────────────
        dn       = self.day_night.predict(frame)
        working  = frame
        enhanced = False

        # ── 2. Enhance if night ───────────────────────────────────────────────
        if dn["route_to_enhancement"]:
            working  = self.enhancer.enhance(frame)
            enhanced = True

        # ── 3. Optional downscale for inference speed ─────────────────────────
        infer_frame = self._maybe_downscale(working)

        # ── 4. Parallel inference (vehicles / persons / faces) ─────────────────
        run_anpr = (
            self._anpr_interval == 0
            or (self._frame_idx % self._anpr_interval) == 1
        )

        fut_vehicles : Future = self._pool.submit(
            self.vehicles.detect, infer_frame, frame_id, self.camera_id
        )
        fut_persons : Future = self._pool.submit(
            self.person_detector.detect, infer_frame, frame_id, self.camera_id
        )
        fut_faces : Future = self._pool.submit(
            self.recogniser.recognise, infer_frame, self.watchlist, frame_id, self.camera_id
        )

        # Collect results (blocks until each is ready)
        vehicle_result : VehicleDetectionResult = fut_vehicles.result()
        person_result  : PersonDetectionResult  = fut_persons.result()
        face_results   : list[FaceResult]       = fut_faces.result()

        # ── 4b. ANPR — cropped to vehicle boxes, run AFTER vehicle detection ───
        # NOTE: adjust `v.bbox` below if VehicleDetection's bbox field is named
        # differently in clashifiers/vechile_detector/main.py (e.g. v.box, v.xyxy).
        if run_anpr:
            vehicle_boxes = [v.bbox for v in vehicle_result.vehicles]
            anpr_result = self.anpr.read_plates_in_vehicles(
                infer_frame, vehicle_boxes, frame_id, self.camera_id
            )
            self._last_anpr_result = anpr_result
        else:
            # Reuse last known result; update its frame_id so the API stays consistent
            anpr_result = self._last_anpr_result
            if anpr_result is not None:
                anpr_result.frame_id = frame_id

        # ── 5. Alerts ─────────────────────────────────────────────────────────
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
import cv2
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Optional

from clashifiers.day_night.main         import DayNightClassifier
from clashifiers.vechile_detector.main   import VehicleDetector, VehicleDetectionResult
from clashifiers.face_recognization.main import FaceRecogniser, DetectedFace
from clashifiers.person_detector.main   import PersonDetector, PersonDetectionResult
from services.zero_dce  import ZeroDCEEnhancer
from services.sci_enhance import SCIEnhancer
from services.anpr      import ANPRService, ANPRResult
from services.dehazing  import DehazingService
from clashifiers.person_reid.main import PersonReIdentifier
from pg_vector import PersonRegistry
from services.ganns import SRGANEnhancer
import logging

logger = logging.getLogger("smart_city_pipeline")
# ─── Frame Event ──────────────────────────────────────────────────────────────

@dataclass
class FrameEvent:
    """
    Single unified event record per frame.

    NOTE: `faces` now holds raw DetectedFace + best_match dicts sourced
    ENTIRELY from PersonRegistry (pgvector) -- the old in-memory Watchlist
    has been removed. Previously the codebase had TWO separate identity
    stores (Watchlist for `faces`/alerts, PersonRegistry for
    `identified_people`) that could disagree with each other on the same
    photo. There is now exactly one source of truth.
    """
    frame_id  : str
    camera_id : str
    timestamp : float
    day_night : dict
    enhanced  : bool                             = False
    vehicles  : Optional[VehicleDetectionResult] = None
    persons   : Optional[PersonDetectionResult]  = None
    plates    : Optional[ANPRResult]             = None
    faces     : list                             = field(default_factory=list)
    identified_people: list                      = field(default_factory=list)
    alerts    : list                             = field(default_factory=list)

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
            "faces":     self.faces,
            "identified_people": self.identified_people,
            "alerts":    self.alerts,
        }


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class SmartCityPipeline:
    def __init__(
        self,
        camera_id           : str   = "cam_01",
        vehicle_model_size  : str   = "yolov8m",
        face_ctx_id         : int   = -1,
        day_night_weights   : Optional[str] = None,
        face_sim_threshold  : float = 0.20,
        body_sim_threshold  : float = 0.22,   # was 0.60 -- see pg_vector.py docstring
        body_match_min_margin: float = 0.06,
        reid_device         : str   = "cpu",
        anpr_interval       : int   = 5,
        inference_max_side  : int   = 960,
        face_det_size       : int   = 320,   # base/fallback size; adaptive_detect
                                              # will scale this up for crowds
        n_threads           : int   = 4,
        plate_model_path    : Optional[str] = None,
        anpr_min_confidence : float = 0.10,
        enhance_gamma        : bool  = True,
        enhance_clahe         : bool  = True,
        enhance_zero_dce      : bool  = True,
        gamma_target_mean     : float = 128.0,
        clahe_clip_limit      : float = 3.0,
        clahe_tile_grid_size  : tuple = (8, 8),
        zero_dce_weights_path : Optional[str] = None,
        zero_dce_scale_factor : int   = 12,
        zero_dce_device        : str   = "cpu",
        night_enhancement_backend : str = "sci",  # NEW — "sci" (default) | "zero_dce"
                                                   # See services/sci_enhance.py docstring for why
                                                   # SCI replaces Zero-DCE++ as the default: it was
                                                   # built and benchmarked specifically for low-light
                                                   # FACE DETECTION as a downstream task, not just
                                                   # perceptual quality, which is what was silently
                                                   # starving PersonDetector/FaceRecogniser at night.
        sci_weights_path       : Optional[str] = None,  # defaults to weights/sci_difficult.pt
                                                          # (DARK FACE-trained) inside sci_enhance.py
        sci_device              : str  = "cpu",
        sci_n_threads           : int  = 4,
        sci_enable_post_clahe   : bool = True,
        face_embedding_history_len : int = 5,     # NEW — frames of face embedding to pool per track
        track_history_prune_after  : int = 150,   # NEW — drop a track's history after this many
                                                   #        frames of not being seen
    ):
        self.camera_id = camera_id
        self._frame_idx = 0

        self.face_sim_threshold = face_sim_threshold
        self.body_sim_threshold = body_sim_threshold
        self.reid_device = reid_device

        self.reidentifier = PersonReIdentifier(device=self.reid_device)
        self.person_registry = PersonRegistry(
           dsn="postgresql://postgres:Quotus%401234@localhost:5432/smart_city",
           face_sim_threshold=self.face_sim_threshold,
           body_sim_threshold=self.body_sim_threshold,
           body_match_min_margin=body_match_min_margin,
        )

        self._anpr_interval    = anpr_interval
        self._last_anpr_result : Optional[ANPRResult] = None
        self._inference_max_side = inference_max_side

        # ── NEW: per-track rolling face embedding history ──────────────────
        # A single frame's ArcFace embedding under night enhancement is noisy
        # enough that a known person can drop below face_sim_threshold on any
        # given frame, even though the face was detected correctly. BotSORT
        # already gives each person a stable track_id, so instead of matching
        # on one frame's embedding, we pool the last N embeddings seen for
        # that track (mean of L2-normalized vectors, renormalized) and match
        # against that. See _update_track_face_history() below.
        self._face_embedding_history_len = face_embedding_history_len
        self._track_history_prune_after  = track_history_prune_after
        self._track_face_embeddings: dict = {}     # track_id -> list[np.ndarray]
        self._track_last_seen_frame: dict = {}      # track_id -> frame_idx

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
            det_size = (face_det_size, face_det_size),
        )
        # NOTE: the in-memory Watchlist has been REMOVED. All face identity
        # matching now goes through self.person_registry (pgvector), which is
        # the same store used for body/person identification below. This is
        # the fix for faces and identified_people disagreeing with each other
        # on the same photo -- there was previously no single source of truth.

        print("[Pipeline] Loading Night Enhancer "
              "(Gamma Correction -> CLAHE -> Zero-DCE++ chain) ...")

        self.night_enhancement_backend = night_enhancement_backend

        if night_enhancement_backend == "sci":
            print("[Pipeline] Loading Night Enhancer "
                  "(SCI — Self-Calibrated Illumination, CVPR2022, DARK FACE-trained) ...")
            sci_kwargs = dict(
                device            = sci_device,
                n_threads         = sci_n_threads,
                enable_post_clahe = sci_enable_post_clahe,
            )
            if sci_weights_path:
                sci_kwargs["weights_path"] = sci_weights_path
            self.enhancer = SCIEnhancer(**sci_kwargs)

        elif night_enhancement_backend == "zero_dce":
            print("[Pipeline] Loading Night Enhancer "
                  "(Gamma Correction -> CLAHE -> Zero-DCE++ chain) ...")
            enhancer_kwargs = dict(
                scale_factor          = zero_dce_scale_factor,
                device                 = zero_dce_device,
                enable_gamma            = enhance_gamma,
                enable_clahe             = enhance_clahe,
                enable_zero_dce          = enhance_zero_dce,
                gamma_target_mean        = gamma_target_mean,
                clahe_clip_limit         = clahe_clip_limit,
                clahe_tile_grid_size     = clahe_tile_grid_size,
            )
            if zero_dce_weights_path:
                enhancer_kwargs["weights_path"] = zero_dce_weights_path
            self.enhancer = ZeroDCEEnhancer(**enhancer_kwargs)

        else:
            raise ValueError(
                f"Unknown night_enhancement_backend={night_enhancement_backend!r}. "
                f"Expected 'sci' or 'zero_dce'."
            )

        print("[Pipeline] Loading Person Detector (YOLOv8) ...")
        self.person_detector = PersonDetector(
            model_size      = vehicle_model_size,
            enable_tracking = True,
        )

        print("[Pipeline] Loading ANPR Service (EasyOCR) ...")
        self.anpr = ANPRService(
            plate_model_path = plate_model_path,
            min_confidence   = anpr_min_confidence,
        )

        print("[Pipeline] Loading Dehazing Service (Dark Channel Prior) ...")
        self.dehazer = DehazingService()

        self._pool = ThreadPoolExecutor(max_workers=n_threads,
                                        thread_name_prefix="sc_infer")

        print(
            f"[Pipeline] All models ready.\n"
            f"           anpr_interval={anpr_interval}  "
            f"inference_max_side={inference_max_side}  "
            f"face_det_size(base)={face_det_size}  "
            f"n_threads={n_threads}  "
            f"body_sim_threshold={body_sim_threshold}  "
            f"face_embedding_history_len={face_embedding_history_len}  "
            f"plate_model={'YOLO (' + plate_model_path + ')' if plate_model_path else 'contour fallback'}\n"
            f"           night_enhancement_backend={night_enhancement_backend}  "
            f"({self.enhancer.method})\n"
        )
        self.sr_enhancer = SRGANEnhancer(
            weights_path='weights/realesr-general-x4v3.pth',      # weights/realesr-general-x4v3.pth
            device="cpu",
            scale=4,    
            min_side_trigger=260,   # skip SR for anything already >= 240p-ish
            target_min_side=480,
            n_threads=4,
)


    def _maybe_downscale(self, frame: np.ndarray) -> np.ndarray:
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

    # ── NEW: track-level face embedding aggregation ────────────────────────


    def _update_track_face_history(self, track_id, embedding: np.ndarray) -> np.ndarray:
        _TRACK_RESET_SIM_THRESHOLD = 0.40  # class-level constant near the top of SmartCityPipeline
        embedding = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        history = self._track_face_embeddings.setdefault(track_id, [])

        # ── Guard against BotSORT track-ID switches ─────────────────────────
        # If this track already has a pooled identity and the new face looks
        # nothing like it, this is very likely a different person who inherited
        # the same track_id after an occlusion/handoff -- reset instead of
        # blending two different people's embeddings together.
        if history:
            prev_stacked = np.stack(history)
            prev_norms = np.linalg.norm(prev_stacked, axis=1, keepdims=True)
            prev_norms[prev_norms == 0] = 1e-6
            prev_aggregated = (prev_stacked / prev_norms).mean(axis=0)
            agg_norm = np.linalg.norm(prev_aggregated)
            if agg_norm > 0:
                prev_aggregated = prev_aggregated / agg_norm

            sim_to_history = float(np.dot(embedding, prev_aggregated))
            if sim_to_history < _TRACK_RESET_SIM_THRESHOLD:
                logger.info(
                    f"track {track_id}: new face sim={sim_to_history:.3f} to existing "
                    f"history is below reset threshold ({_TRACK_RESET_SIM_THRESHOLD}) -- "
                    f"likely a track-ID handoff to a different person; resetting history."
                ) if 'logger' in dir() else None
                history.clear()

        history.append(embedding)
        if len(history) > self._face_embedding_history_len:
            history.pop(0)
        self._track_last_seen_frame[track_id] = self._frame_idx

        stacked = np.stack(history)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        normalized = stacked / norms
        aggregated = normalized.mean(axis=0)
        agg_norm = np.linalg.norm(aggregated)
        if agg_norm > 0:
            aggregated = aggregated / agg_norm
        return aggregated

    def _prune_track_history(self):
        """Drop history for tracks not seen recently, so a long-running
        stream doesn't accumulate unbounded per-track embedding buffers."""
        if not self._track_last_seen_frame:
            return
        stale = [
            tid for tid, last_seen in self._track_last_seen_frame.items()
            if self._frame_idx - last_seen > self._track_history_prune_after
        ]
        for tid in stale:
            self._track_face_embeddings.pop(tid, None)
            self._track_last_seen_frame.pop(tid, None)

    def process_frame(self, frame: np.ndarray) -> "FrameEvent":
        self._frame_idx += 1
        frame_id  = f"frame_{self._frame_idx:06d}"
        timestamp = time.time()
        identified_people = []
        # frame = self.sr_enhancer.enhance(frame)
        logger.info(
            "SR: method=%s triggered=%s scale=%.2f frame_shape=%s",
            self.sr_enhancer.method,
            self.sr_enhancer.last_triggered,
            self.sr_enhancer.last_scale_applied,
            frame.shape,
        )
        self._prune_track_history()

        dn       = self.day_night.predict(frame)
        working  = frame
        enhanced = False

        if dn["route_to_enhancement"]:
            working  = self.enhancer.enhance(frame)
            enhanced = True

        infer_frame = self._maybe_downscale(working)

        run_anpr = (
            self._anpr_interval == 0
            or (self._frame_idx % self._anpr_interval) == 1
        )

        # ── Vehicle + Person detection run in parallel (independent of faces) ──
        fut_vehicles: Future = self._pool.submit(
            self.vehicles.detect, infer_frame, frame_id, self.camera_id
        )
        # Person detection is needed synchronously BEFORE face detection now,
        # because adaptive_detect() needs a person-count hint to pick the
        # right detection grid size.
        person_result: PersonDetectionResult = self.person_detector.adaptive_detect(
            infer_frame,
            frame_id          = frame_id,
            camera_id         = self.camera_id,
            person_count_hint = 0,   # no prior count on first pass; adaptive uses result
        )
        person_count_hint = len(person_result.detections)

        faces: list[DetectedFace] = self.recogniser.adaptive_detect(
            infer_frame, person_count_hint=person_count_hint
        )

        vehicle_result: VehicleDetectionResult = fut_vehicles.result()

        # ── Bind faces to person tracks ONCE, reused by both matching passes ──
        # (previously this spatial-binding loop was duplicated below; doing it
        # once here also lets us build the per-track aggregated embedding
        # before either matching pass runs.)
        track_matching_face = {}   # person track_id -> DetectedFace
        face_query_embedding = {}  # id(face) -> embedding to actually match with (aggregated if bound)

        for person_det in person_result.detections:
            x1, y1, x2, y2 = person_det.bbox
            matching_face = None
            for face in faces:
                fx1, fy1, fx2, fy2 = face.bbox
                face_cx, face_cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                if x1 <= face_cx <= x2 and y1 <= face_cy <= y2:
                    matching_face = face
                    break

            if matching_face is None:
                continue

            track_matching_face[person_det.track_id] = matching_face

            if matching_face.embedding is not None:
                aggregated = self._update_track_face_history(
                    person_det.track_id, matching_face.embedding
                )
                face_query_embedding[id(matching_face)] = aggregated

        # ── Face identity matching — single source of truth: person_registry ──
        # Uses the track-aggregated embedding when this face is bound to a
        # tracked person (smooths night-time flicker); falls back to the raw
        # single-frame embedding for faces with no body/track binding.
        face_dicts = []
        alerts = []
        for face in faces:
            query_embedding = face_query_embedding.get(id(face), face.embedding)
            best_match, sim = self.person_registry.match_face(query_embedding) \
                if query_embedding is not None else (None, 0.0)
            face_dicts.append({
                "bbox": face.bbox,
                "confidence": round(face.confidence, 4),
                "best_match": best_match,
                "similarity": round(sim, 4),
            })
            if best_match is not None:
                alerts.append({
                    "type":       "face_watchlist_hit",
                    "person_id":  best_match["person_id"],
                    "name":       best_match["name"],
                    "image_url":  best_match.get("image_url"),
                    "similarity": sim,
                    "frame_id":   frame_id,
                    "camera_id":  self.camera_id,
                    "timestamp":  timestamp,
                })

        if run_anpr:
            vehicle_boxes = [v.bbox for v in vehicle_result.vehicles]
            anpr_result = self.anpr.read_plates_in_vehicles(
                infer_frame, vehicle_boxes, frame_id, self.camera_id
            )
            self._last_anpr_result = anpr_result
        else:
            anpr_result = self._last_anpr_result
            if anpr_result is not None:
                anpr_result.frame_id = frame_id

        # ── Person Registry Match (body + face-in-body binding) ────────────────
        for person_det in person_result.detections:
            x1, y1, x2, y2 = person_det.bbox
            person_crop = infer_frame[int(y1):int(y2), int(x1):int(x2)]

            if person_crop.size == 0:
                continue

            matching_face = track_matching_face.get(person_det.track_id)

            face_embedding = None
            if matching_face is not None and matching_face.embedding is not None:
                # Use the track-aggregated embedding, same one used above,
                # so /events and identified_people agree with each other.
                face_embedding = face_query_embedding.get(
                    id(matching_face), matching_face.embedding
                )

            # Only compute a body embedding if we don't already have a face --
            # identify() now ignores body_embedding whenever a face was found,
            # so skip the (non-trivial) Re-ID cost entirely in that case.
            body_embedding = None
            crop_shape = None
            if face_embedding is None:
                body_embedding = self.reidentifier.embed(person_crop)
                crop_shape = person_crop.shape[:2]

            identity = self.person_registry.identify(
                face_embedding=face_embedding,
                body_embedding=body_embedding,
                body_crop_shape=crop_shape,
            )

            if identity:
                identified_people.append({
                    "track_id": person_det.track_id,
                    **identity,
                })
                alerts.append({
                    "type": "person_registry_hit",
                    "person_id": identity["person_id"],
                    "name": identity["name"],
                    "image_url": identity.get("image_url"),
                    "similarity": identity["similarity"],
                    "method": identity["method"],
                    "track_id": person_det.track_id,
                    "frame_id": frame_id,
                    "camera_id": self.camera_id,
                    "timestamp": timestamp,
                })

        return FrameEvent(
            frame_id  = frame_id,
            camera_id = self.camera_id,
            timestamp = timestamp,
            day_night = dn,
            enhanced  = enhanced,
            vehicles  = vehicle_result,
            persons   = person_result,
            plates    = anpr_result,
            faces     = face_dicts,
            identified_people = identified_people,
            alerts    = alerts,
        )

    def annotate(self, frame: np.ndarray, event: "FrameEvent") -> np.ndarray:
        out = frame.copy()
        if event.vehicles:
            out = self.vehicles.draw(out, event.vehicles)

        # ── Build a quick lookup: track_id → identity dict ────────────────────
        # identified_people is the authoritative registry result (uses track-
        # aggregated embeddings). We use it to label person body boxes AND to
        # upgrade any face box that was matched here but not in the raw faces loop.
        identity_by_track: dict[int, dict] = {
            p["track_id"]: p for p in event.identified_people if p.get("track_id") is not None
        }

        # ── Draw person body boxes ─────────────────────────────────────────────
        if event.persons:
            for det in event.persons.detections:
                x1, y1, x2, y2 = [int(v) for v in det.bbox]
                identity = identity_by_track.get(det.track_id)
                if identity:
                    # Known person — green box + name label
                    color = (0, 220, 60)
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    method_tag = f"[{identity['method']}]" if identity.get("method") else ""
                    label = f"{identity['name']} {identity['similarity']:.2f} {method_tag}"
                    # Draw a filled background strip so text is readable
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                    cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(out, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)
                else:
                    # Unknown person — dim grey box only (no label noise)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (120, 120, 120), 1)

        # ── Draw face boxes ────────────────────────────────────────────────────
        for f in event.faces:
            x1, y1, x2, y2 = [int(v) for v in f["bbox"]]
            best_match = f["best_match"]
            sim = f["similarity"]

            if best_match is not None:
                # Face directly matched via per-frame embedding
                color = (0, 80, 255)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"{best_match['name']} {sim:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
                cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                cv2.putText(out, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
            else:
                # Unknown face
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 80), 1)
                cv2.putText(out, f"? {f['confidence']:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 80), 1)

        # ── Day/Night + enhancement label ──────────────────────────────────────
        label = f"{event.day_night['label'].upper()}  {event.day_night['confidence']:.2f}"
        if event.enhanced:
            label += "  [ENHANCED]"
        cv2.putText(out, label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return out


    def run_stream(
        self,
        source      : str | int = 0,
        show_window : bool      = True,
        on_event                = None,
    ):
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
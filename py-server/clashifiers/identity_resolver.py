"""
IdentityResolver
-----------------
Path: clashifiers/identity_resolver.py

Centralises per-frame identity resolution so that both the live pipeline
and the /analyse/identify REST endpoint use exactly the same logic.

Core behaviours:
  1. Adaptive face detection grid — 320 for 1-2 people, 640 for 3-8, 960
     for crowds — retrying upward if the first pass finds nothing.
  2. Spatial face↔body binding — a body detection inherits its face's
     identity when the face centre falls inside the body bounding box,
     instead of running a separate Re-ID vote that often picks the wrong person.
  3. Body-only matching disabled by default (enable_body_matching=False) —
     purely body-based Re-ID is noisy in CCTV conditions; turn it on only
     after you've validated it works for your camera placements.

Output schema (returned by identify_frame):
  {
    "people": [
      {
        "track_id":   int | null,
        "body_bbox":  [x1, y1, x2, y2],
        "face_bbox":  [x1, y1, x2, y2] | null,    # null if no face bound
        "person_id":  str | null,
        "name":       str | null,
        "similarity": float | null,
        "method":     "face" | "body" | null,
      },
      ...
    ],
    "unbound_faces": [
      {
        "bbox":        [x1, y1, x2, y2],
        "confidence":  float,
        "person_id":   str | null,
        "name":        str | null,
        "similarity":  float | null,
      },
      ...
    ],
  }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("identity_resolver")


@dataclass
class _BoundPerson:
    """Internal per-frame record before serialisation."""
    track_id:   Optional[int]
    body_bbox:  list[int]
    face_bbox:  Optional[list[int]] = None
    face_conf:  float = 0.0
    face_emb:   Optional[np.ndarray] = None
    body_emb:   Optional[np.ndarray] = None
    # resolved identity
    person_id:  Optional[str] = None
    name:       Optional[str] = None
    similarity: Optional[float] = None
    method:     Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "track_id":   self.track_id,
            "body_bbox":  self.body_bbox,
            "face_bbox":  self.face_bbox,
            "person_id":  self.person_id,
            "name":       self.name,
            "similarity": round(self.similarity, 4) if self.similarity is not None else None,
            "method":     self.method,
        }


class IdentityResolver:
    """
    Stateless per-frame identity resolver.

    Parameters
    ----------
    registry:
        PersonRegistry (pg_vector.PersonRegistry or the JSON-backed fallback).
    face_recogniser:
        FaceRecogniser instance — must implement detect() and optionally
        adaptive_detect().
    person_detector:
        PersonDetector (YOLOv8) for body bounding boxes.
    reidentifier:
        PersonReIdentifier for body Re-ID embeddings.
    enable_body_matching:
        When True, persons without a matched face are identified via Re-ID.
        Disabled by default — body-only Re-ID has a high false-positive rate
        until tuned per deployment.
    body_match_min_crop_px:
        Minimum body crop dimension (px) required before attempting body-only
        matching.  Tiny crops produce unreliable Re-ID embeddings.
    """

    def __init__(
        self,
        registry,
        face_recogniser,
        person_detector,
        reidentifier,
        enable_body_matching: bool = False,
        body_match_min_crop_px: int = 64,
    ):
        self.registry               = registry
        self.face_recogniser        = face_recogniser
        self.person_detector        = person_detector
        self.reidentifier           = reidentifier
        self.enable_body_matching   = enable_body_matching
        self.body_match_min_crop_px = body_match_min_crop_px

    # ---------------------------------------------------------------- #
    # Public API
    # ---------------------------------------------------------------- #

    def identify_frame(
        self,
        frame: np.ndarray,
        frame_id: str = "frame_0",
        camera_id: str = "cam_0",
    ) -> dict:
        """
        Run adaptive face detection + body detection on *frame*, bind faces to
        bodies spatially, resolve identities, and return the structured dict.
        """
        h, w = frame.shape[:2]

        # ── 1. Body detection ─────────────────────────────────────────────────
        det_result = self.person_detector.detect(
            frame, frame_id=frame_id, camera_id=camera_id
        )
        body_detections = getattr(det_result, "detections", [])
        person_count    = len(body_detections)

        # ── 2. Adaptive face detection ────────────────────────────────────────
        if hasattr(self.face_recogniser, "adaptive_detect"):
            raw_faces = self.face_recogniser.adaptive_detect(
                frame, person_count_hint=person_count
            )
        else:
            raw_faces = self.face_recogniser.detect(frame)

        # ── 3. Spatial face→body binding ─────────────────────────────────────
        #  For each body bbox, find a face whose centre falls inside it.
        #  Each face may only bind to one body (greedy, highest-confidence first).
        bound_people: list[_BoundPerson] = []
        unbound_face_idxs = set(range(len(raw_faces)))

        # Sort faces descending by confidence so the best one wins when two
        # faces are near the same body (e.g. two people very close together).
        sorted_face_idxs = sorted(
            range(len(raw_faces)),
            key=lambda i: raw_faces[i].confidence,
            reverse=True,
        )

        for body_det in body_detections:
            bx1, by1, bx2, by2 = body_det.bbox
            bound_face_idx: Optional[int] = None

            for fi in sorted_face_idxs:
                if fi not in unbound_face_idxs:
                    continue  # already bound to another body
                face = raw_faces[fi]
                fx1, fy1, fx2, fy2 = face.bbox
                cx = (fx1 + fx2) / 2
                cy = (fy1 + fy2) / 2
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    bound_face_idx = fi
                    unbound_face_idxs.discard(fi)
                    break

            bp = _BoundPerson(
                track_id  = body_det.track_id,
                body_bbox = body_det.bbox,
            )
            if bound_face_idx is not None:
                face_det         = raw_faces[bound_face_idx]
                bp.face_bbox     = face_det.bbox
                bp.face_conf     = face_det.confidence
                bp.face_emb      = face_det.embedding

            bound_people.append(bp)

        # ── 4. Body Re-ID embeddings (only for body-only matching path) ────────
        if self.enable_body_matching:
            for bp in bound_people:
                if bp.face_emb is not None:
                    continue  # face match will take priority; skip body embed cost
                bx1, by1, bx2, by2 = bp.body_bbox
                crop = frame[int(by1):int(by2), int(bx1):int(bx2)]
                if (crop.size > 0
                        and crop.shape[0] >= self.body_match_min_crop_px
                        and crop.shape[1] >= self.body_match_min_crop_px):
                    bp.body_emb = self.reidentifier.embed(crop)

        # ── 5. Identity resolution ────────────────────────────────────────────
        for bp in bound_people:
            # Face-first: more reliable.
            if bp.face_emb is not None:
                person, sim = self.registry.match_face(bp.face_emb)
                if person is not None:
                    bp.person_id  = person["person_id"]
                    bp.name       = person["name"]
                    bp.similarity = sim
                    bp.method     = "face"
                    continue

            # Body-only fallback (opt-in).
            if self.enable_body_matching and bp.body_emb is not None:
                bx1, by1, bx2, by2 = bp.body_bbox
                crop_h = by2 - by1
                crop_w = bx2 - bx1
                person, sim = self.registry.match_body(
                    bp.body_emb,
                    crop_shape=(crop_h, crop_w),
                )
                if person is not None:
                    bp.person_id  = person["person_id"]
                    bp.name       = person["name"]
                    bp.similarity = sim
                    bp.method     = "body"

        # ── 6. Unbound faces (visible face, no body bbox matched) ─────────────
        unbound_face_records = []
        for fi in unbound_face_idxs:
            face = raw_faces[fi]
            record: dict = {
                "bbox":       face.bbox,
                "confidence": round(face.confidence, 4),
                "person_id":  None,
                "name":       None,
                "similarity": None,
            }
            if face.embedding is not None:
                person, sim = self.registry.match_face(face.embedding)
                if person is not None:
                    record["person_id"]  = person["person_id"]
                    record["name"]       = person["name"]
                    record["similarity"] = round(sim, 4)
            unbound_face_records.append(record)

        return {
            "frame_id":      frame_id,
            "camera_id":     camera_id,
            "person_count":  person_count,
            "people":        [bp.to_dict() for bp in bound_people],
            "unbound_faces": unbound_face_records,
        }

"""
Person Registry — Face + Body Combined Identity Store
------------------------------------------------------
Generalises the old face-only `Watchlist` to support registering a person
from a *mix* of clear-face and full-body/CCTV-style reference photos.

Each registered person can end up with:
  - 0..n face_embeddings  (ArcFace, 512-d)  — only if a face was found in that photo
  - 0..n body_embeddings  (OSNet Re-ID)     — from the detected person bbox in that photo

At least one embedding (face or body) is required per photo for it to count.

Persisted to JSON on disk (mirrors the existing Watchlist.save()/load() pattern).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("person_registry")


@dataclass
class Person:
    person_id: str
    name: str
    face_embeddings: list[np.ndarray] = field(default_factory=list)
    body_embeddings: list[np.ndarray] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "person_id": self.person_id,
            "name": self.name,
            "face_embeddings": [e.tolist() for e in self.face_embeddings],
            "body_embeddings": [e.tolist() for e in self.body_embeddings],
        }

    @staticmethod
    def from_json(d: dict) -> "Person":
        return Person(
            person_id=d["person_id"],
            name=d["name"],
            face_embeddings=[np.array(e, dtype=np.float32) for e in d.get("face_embeddings", [])],
            body_embeddings=[np.array(e, dtype=np.float32) for e in d.get("body_embeddings", [])],
        )


@dataclass
class RegistrationOutcome:
    person_id: str
    name: str
    images_received: int
    face_embeddings_added: int
    body_embeddings_added: int
    images_skipped: int  # images where neither a face nor a body was detected


class PersonRegistry:
    """
    In-memory registry of known people, matched by face (ArcFace) first and
    body appearance (Re-ID) as a fallback. Brute-force cosine search —
    for large deployments swap in FAISS/Milvus, same note as the old Watchlist.
    """

    def __init__(
        self,
        face_sim_threshold: float = 0.55,
        body_sim_threshold: float = 0.65,
        storage_path: str | None = None,
    ):
        self.face_sim_threshold = face_sim_threshold
        self.body_sim_threshold = body_sim_threshold
        self.storage_path = storage_path
        self._people: dict[str, Person] = {}

        if storage_path:
            self.load(storage_path)

    # ---------------------------------------------------------------- #
    # Registration
    # ---------------------------------------------------------------- #

    def add_person(
        self,
        name: str,
        images: list[np.ndarray],
        face_recogniser,   # FaceRecogniser instance (for detection + ArcFace embedding)
        person_detector,   # PersonDetector instance (for body bbox)
        reidentifier,      # PersonReIdentifier instance (for body embedding)
        person_id: str | None = None,
    ) -> RegistrationOutcome:
        """
        Registers a person from one or more reference images. Each image is
        processed independently: try to find a face -> ArcFace embed it;
        try to find a body bbox -> Re-ID embed it. An image contributes
        whatever it can (face only, body only, both, or neither).
        """
        pid = person_id or str(uuid.uuid4())
        person = self._people.get(pid, Person(person_id=pid, name=name))
        person.name = name

        face_added = 0
        body_added = 0
        skipped = 0

        for idx, img in enumerate(images):
            found_something = False

            # --- Face path ---
            try:
                faces = face_recogniser.detect_and_embed(img)  # expects list[DetectedFace]-like
                if faces:
                    # take the largest/most confident face in the reference photo
                    best_face = max(faces, key=lambda f: f.confidence)
                    if best_face.embedding is not None:
                        person.face_embeddings.append(np.asarray(best_face.embedding, dtype=np.float32))
                        face_added += 1
                        found_something = True
            except Exception as e:
                logger.warning(f"add_person: face path failed on image {idx}: {e}")

            # --- Body path ---
            try:
                det_result = person_detector.detect(img, frame_id=f"reg_{idx}", camera_id="registration")
                detections = getattr(det_result, "detections", [])
                if detections:
                    best = max(detections, key=lambda d: d.area)
                    x1, y1, x2, y2 = best.bbox
                    crop = img[int(y1):int(y2), int(x1):int(x2)]
                    vec = reidentifier.embed(crop)
                    if vec is not None:
                        person.body_embeddings.append(vec)
                        body_added += 1
                        found_something = True
            except Exception as e:
                logger.warning(f"add_person: body path failed on image {idx}: {e}")

            if not found_something:
                skipped += 1

        self._people[pid] = person
        if self.storage_path:
            self.save(self.storage_path)

        return RegistrationOutcome(
            person_id=pid,
            name=name,
            images_received=len(images),
            face_embeddings_added=face_added,
            body_embeddings_added=body_added,
            images_skipped=skipped,
        )

    def remove(self, person_id: str) -> bool:
        existed = person_id in self._people
        self._people.pop(person_id, None)
        if existed and self.storage_path:
            self.save(self.storage_path)
        return existed

    def list_people(self) -> list[dict]:
        return [
            {
                "person_id": p.person_id,
                "name": p.name,
                "face_refs": len(p.face_embeddings),
                "body_refs": len(p.body_embeddings),
            }
            for p in self._people.values()
        ]

    # ---------------------------------------------------------------- #
    # Matching
    # ---------------------------------------------------------------- #

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def match_face(self, embedding: np.ndarray):
        """Returns (person, similarity) for the best face match, or (None, 0.0)."""
        best_person, best_sim = None, 0.0
        for person in self._people.values():
            for ref in person.face_embeddings:
                sim = self._cosine(embedding, ref)
                if sim > best_sim:
                    best_sim, best_person = sim, person
        if best_person is not None and best_sim >= self.face_sim_threshold:
            return best_person, best_sim
        return None, best_sim

    def match_body(self, embedding: np.ndarray):
        """Returns (person, similarity) for the best body match, or (None, 0.0)."""
        best_person, best_sim = None, 0.0
        for person in self._people.values():
            for ref in person.body_embeddings:
                sim = self._cosine(embedding, ref)
                if sim > best_sim:
                    best_sim, best_person = sim, person
        if best_person is not None and best_sim >= self.body_sim_threshold:
            return best_person, best_sim
        return None, best_sim

    def identify(self, face_embedding: np.ndarray | None, body_embedding: np.ndarray | None) -> dict:
        """
        Combined identification: try face first (more reliable), fall back to body.
        Returns a dict suitable for attaching to a detection, or None if no match.
        """
        if face_embedding is not None:
            person, sim = self.match_face(face_embedding)
            if person is not None:
                return {
                    "person_id": person.person_id,
                    "name": person.name,
                    "similarity": sim,
                    "method": "face",
                }

        if body_embedding is not None:
            person, sim = self.match_body(body_embedding)
            if person is not None:
                return {
                    "person_id": person.person_id,
                    "name": person.name,
                    "similarity": sim,
                    "method": "body",
                }

        return None

    # ---------------------------------------------------------------- #
    # Persistence
    # ---------------------------------------------------------------- #

    def save(self, path: str):
        data = {"people": [p.to_json() for p in self._people.values()]}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._people = {
                p["person_id"]: Person.from_json(p) for p in data.get("people", [])
            }
        except FileNotFoundError:
            self._people = {}
        except Exception as e:
            logger.warning(f"PersonRegistry: failed to load {path}: {e}")
            self._people = {}
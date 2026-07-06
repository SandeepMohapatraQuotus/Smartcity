"""
Person Registry — pgvector-backed version
--------------------------------------------
Same public API as the plain-Postgres version (add_person, remove,
list_people, identify, match_face, match_body) but matching now happens
inside Postgres using pgvector's cosine-distance operator (`<=>`), which
is much faster and index-able (IVFFlat) as the registry grows.

Requires:
    pip install psycopg2-binary pgvector
    Run pgvector_setup.sql once against your target database first.

Note: pgvector's `<=>` returns COSINE DISTANCE (0 = identical), so
similarity = 1 - distance.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

logger = logging.getLogger("person_registry_pgvector")


@dataclass
class RegistrationOutcome:
    person_id: str
    name: str
    images_received: int
    face_embeddings_added: int
    body_embeddings_added: int
    images_skipped: int
    errors: list[str] = None   # per-image failure reasons
    registry_unavailable: bool = False

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class PersonRegistry:
    def __init__(
        self,
        dsn: str,
        face_sim_threshold: float = 0.70,
        body_sim_threshold: float = 0.60,
    ):
        self.dsn = dsn
        self.face_sim_threshold = face_sim_threshold
        self.body_sim_threshold = body_sim_threshold
        self._conn = None
        self._available = False

        try:
            self._conn = psycopg2.connect(dsn)
            self._conn.autocommit = True
            register_vector(self._conn)  # teaches psycopg2 to adapt numpy arrays <-> vector
            self._available = True
            logger.info("PersonRegistry connected to pgvector database.")
        except Exception as e:
            logger.warning(
                f"PersonRegistry: could not connect or enable pgvector ({e}). "
                "Person registry matching will be DISABLED until the database is ready. "
                "Fix: run  sudo apt install postgresql-17-pgvector  then re-run pgvector_setup.sql and restart the server."
            )

    # ---------------------------------------------------------------- #
    # Registration
    # ---------------------------------------------------------------- #

    def add_person(
        self,
        name: str,
        images: list[np.ndarray],
        face_recogniser,
        person_detector,
        reidentifier,
        person_id: str | None = None,
    ) -> RegistrationOutcome:
        if not self._available:
            msg = ("PersonRegistry is unavailable — pgvector extension not installed on the "
                   "PostgreSQL server. Run: sudo apt install postgresql-17-pgvector "
                   "then re-run pgvector_setup.sql and restart the server.")
            logger.warning(msg)
            return RegistrationOutcome(
                person_id=str(uuid.uuid4()), name=name,
                images_received=len(images), face_embeddings_added=0,
                body_embeddings_added=0, images_skipped=len(images),
                registry_unavailable=True,
                errors=[msg],
            )
        pid = person_id or str(uuid.uuid4())

        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO persons (person_id, name) VALUES (%s, %s)
                ON CONFLICT (person_id) DO UPDATE SET name = EXCLUDED.name
                """,
                (pid, name),
            )

        face_added = 0
        body_added = 0
        skipped = 0
        errors: list[str] = []

        for idx, img in enumerate(images):
            found_something = False

            # --- Face path ---
            try:
                faces = face_recogniser.detect(img)
                if faces:
                    best_face = max(faces, key=lambda f: f.confidence)
                    if best_face.embedding is not None:
                        self._insert_embedding("face_embeddings", pid, best_face.embedding)
                        face_added += 1
                        found_something = True
                    else:
                        errors.append(f"img[{idx}] face: embedding extraction returned None")
                else:
                    errors.append(f"img[{idx}] face: no face detected")
            except Exception as e:
                err = f"img[{idx}] face: {type(e).__name__}: {e}"
                errors.append(err)
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
                        self._insert_embedding("body_embeddings", pid, vec)
                        body_added += 1
                        found_something = True
                    else:
                        errors.append(f"img[{idx}] body: Re-ID embed returned None (model unavailable?)")
                else:
                    errors.append(f"img[{idx}] body: no person detected in image")
            except Exception as e:
                err = f"img[{idx}] body: {type(e).__name__}: {e}"
                errors.append(err)
                logger.warning(f"add_person: body path failed on image {idx}: {e}")

            if not found_something:
                skipped += 1

        return RegistrationOutcome(
            person_id=pid,
            name=name,
            images_received=len(images),
            face_embeddings_added=face_added,
            body_embeddings_added=body_added,
            images_skipped=skipped,
            errors=errors,
        )

    def _insert_embedding(self, table: str, person_id: str, vec: np.ndarray):
        vec = np.asarray(vec, dtype=np.float32)
        with self._conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table} (person_id, embedding) VALUES (%s, %s)",
                (person_id, vec),
            )

    def remove(self, person_id: str) -> bool:
        if not self._available:
            return False
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM persons WHERE person_id = %s", (person_id,))
            deleted = cur.rowcount > 0
        return deleted

    def list_people(self) -> list[dict]:
        if not self._available:
            return []
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.person_id, p.name,
                       COUNT(DISTINCT f.id) AS face_refs,
                       COUNT(DISTINCT b.id) AS body_refs
                FROM persons p
                LEFT JOIN face_embeddings f ON f.person_id = p.person_id
                LEFT JOIN body_embeddings b ON b.person_id = p.person_id
                GROUP BY p.person_id, p.name
                """
            )
            return [dict(row) for row in cur.fetchall()]

    # ---------------------------------------------------------------- #
    # Matching — pushed into SQL via pgvector's cosine distance operator
    # ---------------------------------------------------------------- #

    def _match(self, table: str, embedding: np.ndarray, threshold: float):
        if not self._available:
            return None, 0.0
        embedding = np.asarray(embedding, dtype=np.float32)
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT p.person_id, p.name,
                    1 - (e.embedding <=> %s) AS similarity
                FROM {table} e
                JOIN persons p ON p.person_id = e.person_id
                ORDER BY e.embedding <=> %s
                LIMIT 1
                """,
                (embedding, embedding),
            )
            row = cur.fetchone()

        if row is None:
            return None, 0.0
        similarity = float(row["similarity"])
        if similarity >= threshold:
            return {"person_id": row["person_id"], "name": row["name"]}, similarity
        return None, similarity
    def match_face(self, embedding: np.ndarray):
        return self._match("face_embeddings", embedding, self.face_sim_threshold)

    def match_body(self, embedding: np.ndarray):
        return self._match("body_embeddings", embedding, self.body_sim_threshold)

    def identify(self, face_embedding: np.ndarray | None, body_embedding: np.ndarray | None) -> dict | None:
        if face_embedding is not None:
            person, sim = self.match_face(face_embedding)
            if person is not None:
                return {**person, "similarity": sim, "method": "face"}

        if body_embedding is not None:
            person, sim = self.match_body(body_embedding)
            if person is not None:
                return {**person, "similarity": sim, "method": "body"}

        return None

    def close(self):
        if self._conn:
            self._conn.close()
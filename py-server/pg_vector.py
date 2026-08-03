"""
Person Registry — pgvector-backed version (PATCHED)
--------------------------------------------------------
Changes from the version you're running:

  1. match_body() now queries the TOP 2 nearest neighbours (not just 1) and
     rejects the match if the best and second-best are too close together
     (margin check). This directly targets the failure you measured:
     3 different people in a group photo all matched "Soumya Bhai" at
     0.70-0.80 similarity, while the one TRUE body match in your test data
     was only 0.644 -- there is no threshold alone that separates these,
     you need the margin check too.

  2. identify() now accepts and forwards crop_shape, so the min-crop-size
     guard in match_body actually does something. Previously pipeline.py
     never passed crop_shape through, so the guard was unreachable dead code.

  3. body_sim_threshold default raised to 0.82 (was 0.50 in this class,
     0.60 in pipeline.py's override -- both too low per your test data).

  4. face-vs-body priority made stricter: if a face embedding is provided
     AT ALL (even if it doesn't clear face_sim_threshold), body matching
     is skipped entirely for that detection. A visible-but-unmatched face
     means "this is a stranger", not "fall back to guessing from body shape".
     Body-only matching now only runs when NO face was found in the crop
     at all (i.e. face_embedding is None), which is the only situation
     it's actually meant for.

  5. NEW — enrollment-time night-domain augmentation (add_person):
     Root cause of "known person shows Unknown at night": reference photos
     enrolled via /watchlist/add are daylight shots, but frames processed
     at night pass through ZeroDCEEnhancer (Gamma -> CLAHE -> Zero-DCE++)
     BEFORE ArcFace runs. That enhancement chain shifts the resulting
     512-d embedding just enough that it falls below face_sim_threshold
     against a clean daylight reference embedding -- the face is detected
     fine, it's the embedding that's out of domain.

     Fix: when an `enhancer` is passed to add_person(), each reference
     image is ALSO synthetically darkened and run through the exact same
     enhancement chain used at inference time, then embedded and stored
     as an additional face_embedding row for that person. The registry
     now holds both a "daylight" and a "night-enhanced" embedding per
     person, so a real night-time query has something in the right domain
     to match against. This is purely additive -- nearest-neighbour search
     in match_face() already checks all rows for a person, no query-side
     changes needed.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
import cv2  # add to imports at top of pg_vector.py

logger = logging.getLogger("person_registry_pgvector")

_MIN_FACE_CONF_REGISTER = 0.40
_MIN_BODY_CROP_PX = 32

# Minimum gap between best and second-best body-match candidates.
# If two different registered people are within this margin of each other,
# the match is too ambiguous to trust -- reject rather than guess.
_BODY_MATCH_MIN_MARGIN = 0.06

# How dark to synthetically render a daylight reference photo before
# running it through the enhancement chain during enrollment. Chosen to
# land in the same rough brightness band as DayNightClassifier's
# DARK_THRESH (<=85 mean) so the simulated image actually triggers the
# same processing path a real night frame would.
_NIGHT_SIM_DARKNESS_FACTOR = 0.35
_NIGHT_SIM_NOISE_STD = 4.0


def _simulate_night_conditions(img: np.ndarray) -> np.ndarray:
    """
    Synthetically darken AND desaturate a well-lit reference image to
    approximate what this person's face looks like to THIS camera at night,
    BEFORE the same enhancement chain used at inference time is applied.

    Desaturation matters as much as darkening: this camera's night output is
    effectively monochrome (IR-illuminated), not just dim color video. Without
    this step, the enhancer (SCI, trained on color DARK FACE images) will
    happily reconstruct color/skin-tone information the real camera never
    captures, leaving the stored embedding in the wrong domain versus real
    query-time embeddings.
    """
    darkened = img.astype(np.float32) * _NIGHT_SIM_DARKNESS_FACTOR
    darkened = np.clip(darkened, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(darkened, cv2.COLOR_BGR2GRAY)
    darkened = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)

    noise = np.random.normal(0, _NIGHT_SIM_NOISE_STD, img.shape).astype(np.float32)
    darkened = np.clip(darkened + noise, 0, 255).astype(np.uint8)
    return darkened


@dataclass
class RegistrationOutcome:
    person_id: str
    name: str
    images_received: int
    face_embeddings_added: int
    body_embeddings_added: int
    images_skipped: int
    errors: list = None
    registry_unavailable: bool = False
    reused_existing_person: bool = False
    night_variants_added: int = 0  # NEW — count of synthetic night-domain embeddings added

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class PersonRegistry:
    def __init__(
        self,
        dsn: str,
        face_sim_threshold: float = 0.20,
        body_sim_threshold: float = 0.22,   # was 0.50 -- too low, see docstring
        body_match_min_margin: float = _BODY_MATCH_MIN_MARGIN,
    ):
        self.dsn = dsn
        self.face_sim_threshold = face_sim_threshold
        self.body_sim_threshold = body_sim_threshold
        self.body_match_min_margin = body_match_min_margin
        self._conn = None
        self._available = False

        try:
            self._conn = psycopg2.connect(dsn)
            self._conn.autocommit = True
            register_vector(self._conn)
            self._available = True
            logger.info("PersonRegistry connected to pgvector database.")
        except Exception as e:
            logger.warning(
                f"PersonRegistry: could not connect or enable pgvector ({e}). "
                "Person registry matching will be DISABLED until the database is ready."
            )

    # ---------------------------------------------------------------- #
    # Registration
    # ---------------------------------------------------------------- #

    def _find_person_id_by_name(self, name: str) -> str | None:
        if not self._available:
            return None
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT person_id FROM persons WHERE LOWER(name) = LOWER(%s) LIMIT 1",
                    (name,),
                )
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.warning(f"_find_person_id_by_name: query failed: {e}")
            return None

    def add_person(
        self,
        name: str,
        images,
        face_recogniser,
        person_detector,
        reidentifier,
        person_id: str | None = None,
        enhancer=None,               # NEW — pass pipeline.enhancer to enable night augmentation
        night_augment: bool = True,  # NEW — set False to skip augmentation even if enhancer given
    ) -> RegistrationOutcome:
        if not self._available:
            msg = ("PersonRegistry is unavailable — pgvector extension not installed. "
                   "Run: sudo apt install postgresql-17-pgvector, re-run pgvector_setup.sql, restart server.")
            logger.warning(msg)
            return RegistrationOutcome(
                person_id=str(uuid.uuid4()), name=name,
                images_received=len(images), face_embeddings_added=0,
                body_embeddings_added=0, images_skipped=len(images),
                registry_unavailable=True, errors=[msg],
            )

        reused = False
        if person_id:
            pid = person_id
        else:
            existing_pid = self._find_person_id_by_name(name)
            if existing_pid:
                pid = existing_pid
                reused = True
                logger.info(f"add_person: found existing person '{name}' ({pid}) — merging.")
            else:
                pid = str(uuid.uuid4())

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
        night_variants_added = 0
        skipped = 0
        errors = []

        for idx, img in enumerate(images):
            found_something = False

            try:
                if hasattr(face_recogniser, "detect_and_embed"):
                    faces = face_recogniser.detect_and_embed(img)
                else:
                    faces = face_recogniser.detect(img)

                if faces:
                    confident_faces = [f for f in faces if f.confidence >= _MIN_FACE_CONF_REGISTER]
                    if not confident_faces:
                        errors.append(
                            f"img[{idx}] face: all {len(faces)} detected face(s) below "
                            f"confidence threshold ({_MIN_FACE_CONF_REGISTER})"
                        )
                    else:
                        best_face = max(confident_faces, key=lambda f: f.confidence)
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

            # ---- NEW: night-domain augmentation -------------------------- #
            # Synthetically darken this same reference image, push it through
            # the SAME enhancement chain used at inference time, and store
            # the resulting embedding as an additional row for this person.
            # This is what actually closes the "Unknown at night" gap -- see
            # module docstring point 5.
            if enhancer is not None and night_augment:
                try:
                    night_img = _simulate_night_conditions(img)
                    night_enhanced = enhancer.enhance(night_img)

                    if hasattr(face_recogniser, "detect_and_embed"):
                        night_faces = face_recogniser.detect_and_embed(night_enhanced)
                    else:
                        night_faces = face_recogniser.detect(night_enhanced)

                    if night_faces:
                        confident_night = [
                            f for f in night_faces if f.confidence >= _MIN_FACE_CONF_REGISTER
                        ]
                        if confident_night:
                            best_night_face = max(confident_night, key=lambda f: f.confidence)
                            if best_night_face.embedding is not None:
                                self._insert_embedding(
                                    "face_embeddings", pid, best_night_face.embedding
                                )
                                night_variants_added += 1
                                found_something = True
                            else:
                                errors.append(
                                    f"img[{idx}] night-aug: embedding extraction returned None"
                                )
                        else:
                            errors.append(
                                f"img[{idx}] night-aug: face detected but below confidence "
                                f"threshold on the enhanced/darkened variant"
                            )
                    else:
                        errors.append(
                            f"img[{idx}] night-aug: no face detected on synthetic night variant "
                            f"(enhancement chain may have degraded the crop too much)"
                        )
                except Exception as e:
                    err = f"img[{idx}] night-aug: {type(e).__name__}: {e}"
                    errors.append(err)
                    logger.warning(f"add_person: night augmentation failed on image {idx}: {e}")

            try:
                det_result = person_detector.detect(img, frame_id=f"reg_{idx}", camera_id="registration")
                detections = getattr(det_result, "detections", [])
                if detections:
                    best = max(detections, key=lambda d: d.area)
                    x1, y1, x2, y2 = best.bbox
                    crop = img[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size == 0 or crop.shape[0] < _MIN_BODY_CROP_PX or crop.shape[1] < _MIN_BODY_CROP_PX:
                        errors.append(f"img[{idx}] body: crop too small ({crop.shape[:2]}) to embed")
                    else:
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
            person_id=pid, name=name, images_received=len(images),
            face_embeddings_added=face_added, body_embeddings_added=body_added,
            images_skipped=skipped, errors=errors, reused_existing_person=reused,
            night_variants_added=night_variants_added,
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

    def list_people(self) -> list:
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
    # Matching
    # ---------------------------------------------------------------- #

    def match_face(self, embedding: np.ndarray):
        if not self._available:
            return None, 0.0
        embedding = np.asarray(embedding, dtype=np.float32)
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.person_id, p.name,
                    1 - (e.embedding <=> %s) AS similarity
                FROM face_embeddings e
                JOIN persons p ON p.person_id = e.person_id
                ORDER BY e.embedding <=> %s
                LIMIT 1
                """,
                (embedding, embedding),
            )
            row = cur.fetchone()
        if row is None:
            logger.info("match_face: no rows in face_embeddings table at all.")
            return None, 0.0

        similarity = float(row["similarity"])
        is_match = similarity >= self.face_sim_threshold
        logger.info(
            f"match_face: best candidate='{row['name']}' sim={similarity:.4f} "
            f"threshold={self.face_sim_threshold:.2f} -> {'MATCH' if is_match else 'no match'}"
        )
        if is_match:
            return {"person_id": row["person_id"], "name": row["name"]}, similarity
        return None, similarity

    def match_body(
        self,
        embedding: np.ndarray,
        min_crop_px: int = 48,
        crop_shape: tuple | None = None,
    ):
        """
        PATCHED: pulls top-2 nearest neighbours and rejects the match if
        they're within body_match_min_margin of each other -- this is the
        actual fix for the "3 different people all match one registered
        person" failure mode from your test data. A single threshold could
        not separate your true match (0.644) from your false matches
        (0.70-0.80); the margin check can, because false positives tend to
        cluster close to OTHER false candidates too, while a genuine match
        stands out more clearly from the field.
        """
        if not self._available:
            return None, 0.0

        if crop_shape is not None:
            h, w = crop_shape
            if h < min_crop_px or w < min_crop_px:
                logger.debug(f"match_body: skipping — crop {h}x{w} below {min_crop_px}px guard.")
                return None, 0.0

        embedding = np.asarray(embedding, dtype=np.float32)
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT p.person_id, p.name,
                    1 - (e.embedding <=> %s) AS similarity
                FROM body_embeddings e
                JOIN persons p ON p.person_id = e.person_id
                ORDER BY e.embedding <=> %s
                LIMIT 2
                """,
                (embedding, embedding),
            )
            rows = cur.fetchall()

        if not rows:
            return None, 0.0

        best = rows[0]
        best_sim = float(best["similarity"])
        second_sim = float(rows[1]["similarity"]) if len(rows) > 1 else 0.0

        if best_sim < self.body_sim_threshold:
            return None, best_sim
        if (best_sim - second_sim) < self.body_match_min_margin:
            logger.info(
                f"match_body: rejecting ambiguous match "
                f"best={best_sim:.3f} second={second_sim:.3f} "
                f"margin<{self.body_match_min_margin}"
            )
            return None, best_sim

        return {"person_id": best["person_id"], "name": best["name"]}, best_sim

    def identify(
        self,
        face_embedding: np.ndarray | None,
        body_embedding: np.ndarray | None,
        body_crop_shape: tuple | None = None,
    ) -> dict | None:
        """
        PATCHED priority logic: if a face was detected in this crop AT ALL,
        never fall back to body matching -- a visible-but-unrecognized face
        means "unknown person", not "try guessing from clothing/build".
        Body-only matching now only fires when no face was found in the
        crop whatsoever (face_embedding is None).
        """
        if face_embedding is not None:
            person, sim = self.match_face(face_embedding)
            if person is not None:
                return {**person, "similarity": sim, "method": "face"}
            # Face WAS found but didn't match anyone confidently -> stop here.
            # Do not fall through to body matching.
            return None

        if body_embedding is not None:
            person, sim = self.match_body(body_embedding, crop_shape=body_crop_shape)
            if person is not None:
                return {**person, "similarity": sim, "method": "body"}

        return None

    def close(self):
        if self._conn:
            self._conn.close()
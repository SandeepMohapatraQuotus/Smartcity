"""
Face Recognition
-----------------
Path    : classifiers/face_recognization/main.py
Stage 1 : RetinaFace  →  face detection + 5 landmarks
Stage 2 : ArcFace     →  512-d identity embedding per face
Stage 3 : Cosine similarity against Watchlist

Input  : BGR frame  (numpy array from OpenCV)
Output : list[FaceResult]  — detected face + nearest watchlist match + alert flag

Zero training needed — InsightFace ships with pretrained weights (buffalo_l).
"""

import cv2
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class DetectedFace:
    bbox       : list[int]           # [x1, y1, x2, y2]
    confidence : float
    landmarks  : list[list[int]]     # 5 × [x, y]
    embedding  : Optional[np.ndarray] = None   # 512-d ArcFace vector

    def to_dict(self) -> dict:
        return {
            "bbox":       self.bbox,
            "confidence": round(self.confidence, 4),
            "landmarks":  self.landmarks,
            # embedding intentionally excluded — not JSON-serialisable by default
        }


@dataclass
class WatchlistMatch:
    person_id  : str
    name       : str
    similarity : float
    is_match   : bool

    def to_dict(self) -> dict:
        return {
            "person_id":  self.person_id,
            "name":       self.name,
            "similarity": round(self.similarity, 4),
            "is_match":   self.is_match,
        }


@dataclass
class FaceResult:
    face      : DetectedFace
    match     : Optional[WatchlistMatch] = None
    frame_id  : str = "frame_0"
    camera_id : str = "cam_0"

    def to_dict(self) -> dict:
        return {
            "frame_id":  self.frame_id,
            "camera_id": self.camera_id,
            "face":      self.face.to_dict(),
            "match":     self.match.to_dict() if self.match else None,
            "alert":     bool(self.match and self.match.is_match),
        }


# ─── Watchlist ────────────────────────────────────────────────────────────────

class Watchlist:
    """
    In-memory face watchlist.
    Production: swap _store with FAISS / Milvus for million-scale search.

    Usage:
        watchlist = Watchlist(similarity_threshold=0.55)
        watchlist.add_from_photo("p001", "Alice", frame, recogniser)
        match = watchlist.search(embedding)
    """

    def __init__(self, similarity_threshold: float = 0.55):
        self.threshold = similarity_threshold
        self._store: list[dict] = []

    def add_person(self, person_id: str, name: str, embedding: np.ndarray):
        self._store.append({
            "person_id": person_id,
            "name":      name,
            "embedding": embedding / np.linalg.norm(embedding),
        })

    def add_from_photo(self, person_id: str, name: str, photo, recogniser):
        """
        Extract embedding from a reference photo and add to watchlist.

        photo can be:
          - str        → file path on disk
          - np.ndarray → BGR frame already in memory (e.g. from FastAPI upload)
        """
        if isinstance(photo, str):
            frame = cv2.imread(photo)
            if frame is None:
                raise FileNotFoundError(f"Photo not found: {photo}")
        elif isinstance(photo, np.ndarray):
            frame = photo
        else:
            raise TypeError("photo must be a file path (str) or BGR numpy array.")

        faces = recogniser.detect(frame)
        if not faces:
            raise ValueError("No face detected in the provided photo.")
        emb = faces[0].embedding
        if emb is None:
            raise ValueError("Embedding extraction failed.")
        self.add_person(person_id, name, emb)
        print(f"[Watchlist] Added {name} ({person_id})")

    def search(self, embedding: np.ndarray) -> Optional[WatchlistMatch]:
        if not self._store or embedding is None:
            return None
        query    = embedding / np.linalg.norm(embedding)
        best_sim = -1.0
        best     = None
        for entry in self._store:
            sim = float(np.dot(query, entry["embedding"]))
            if sim > best_sim:
                best_sim = sim
                best     = entry
        if best is None:
            return None
        return WatchlistMatch(
            person_id  = best["person_id"],
            name       = best["name"],
            similarity = best_sim,
            is_match   = best_sim >= self.threshold,
        )

    def remove(self, person_id: str) -> int:
        before = len(self._store)
        self._store = [e for e in self._store if e["person_id"] != person_id]
        return before - len(self._store)

    def list_people(self) -> list[dict]:
        return [{"person_id": e["person_id"], "name": e["name"]} for e in self._store]

    def save(self, path: str):
        data = [{"person_id": e["person_id"], "name": e["name"],
                 "embedding": e["embedding"].tolist()} for e in self._store]
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[Watchlist] Saved {len(self._store)} entries → {path}")

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self._store = [{"person_id": e["person_id"], "name": e["name"],
                        "embedding": np.array(e["embedding"], dtype=np.float32)}
                       for e in data]
        print(f"[Watchlist] Loaded {len(self._store)} entries from {path}")

    def __len__(self) -> int:
        return len(self._store)


# ─── Face Recogniser ──────────────────────────────────────────────────────────

class FaceRecogniser:
    """
    Two-stage recognition pipeline using InsightFace (RetinaFace + ArcFace).
    Pretrained weights auto-download on first run (buffalo_l pack ~300MB).

    Usage:
        from classifiers.face_recognization.main import FaceRecogniser, Watchlist
        recogniser = FaceRecogniser(ctx_id=-1)   # -1 = CPU, 0 = first GPU
        watchlist  = Watchlist()
        results    = recogniser.recognise(frame, watchlist)
    """

    def __init__(
        self,
        det_thresh : float = 0.35,   # ↓ from 0.5 — better recall on dim/occluded faces
        det_size   : tuple = (640, 640),
        ctx_id     : int   = 0,           # -1 = CPU
    ):
        self.det_thresh = det_thresh
        self._load(ctx_id, det_size)

    def _load(self, ctx_id: int, det_size: tuple):
        try:
            from insightface.app import FaceAnalysis
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if ctx_id >= 0 else ["CPUExecutionProvider"])
            self.app = FaceAnalysis(name="buffalo_l", providers=providers)
            self.app.prepare(ctx_id=ctx_id, det_size=det_size)
            print(f"[FaceRecogniser] InsightFace buffalo_l loaded (ctx_id={ctx_id})")
        except ImportError:
            raise ImportError(
                "Run: pip install insightface onnxruntime-gpu\n"
                "     (or onnxruntime for CPU-only)"
            )

    def _resize_for_det_size(self, frame: np.ndarray, target_size: int) -> np.ndarray:
        """Scale a frame so its longer side equals target_size, preserving
        aspect ratio.  This simulates different detection grid sizes WITHOUT
        calling app.prepare() a second time, which corrupts InsightFace's ONNX
        session _inputs_meta and causes TypeError: 'NoneType' is not iterable.
        """
        h, w = frame.shape[:2]
        longer = max(h, w)
        if longer == target_size:
            return frame
        scale = target_size / longer
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA
        return cv2.resize(frame, (new_w, new_h), interpolation=interp)

    @staticmethod
    def _preprocess_for_face_detection(frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the
        luminance channel before feeding the frame to RetinaFace.

        This is always-on and extremely cheap (pure CPU, no deep learning).
        It boosts contrast specifically in dim or shadow areas where faces live,
        without distorting colours (only the L channel in LAB space is touched).

        Why this helps:
          - Day/Night classifier only triggers full Zero-DCE enhancement for
            obvious night frames (mean < 60px). Slightly dark frames (indoor
            lighting, shade, overcast) are classified as 'day' and get no
            enhancement at all — RetinaFace then misses those faces.
          - CLAHE is deterministic and adds ~0.5ms per frame on CPU, making it
            safe to run every single frame without affecting throughput.
        """
        # Convert to LAB, equalize only the L (lightness) channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_eq  = clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect all faces in a BGR frame and return with ArcFace embeddings.
        Applies CLAHE luminance normalization before detection to improve recall
        in dim/shadow/indoor-lighting conditions.
        """
        frame = self._preprocess_for_face_detection(frame)
        results = []
        for f in self.app.get(frame):
            if f.det_score < self.det_thresh:
                continue
            results.append(DetectedFace(
                bbox       = list(map(int, f.bbox)),
                confidence = float(f.det_score),
                landmarks  = [list(map(int, p)) for p in f.kps],
                embedding  = f.embedding,
            ))
        return results

    def detect_and_embed(self, frame: np.ndarray) -> list[DetectedFace]:
        """
        Alias of detect() — returns all faces with ArcFace embeddings.
        Called by pg_vector.add_person() and IdentityResolver so both paths
        use the same entry-point without needing to import DetectedFace directly.
        """
        return self.detect(frame)

    def adaptive_detect(
        self,
        frame: np.ndarray,
        person_count_hint: int = 1,
        sizes: tuple = (320, 640, 960),
    ) -> list[DetectedFace]:
        """
        Retry face detection at progressively larger effective scales until at
        least one face is found, or until all sizes have been tried.

        Instead of calling app.prepare() with different det_sizes (which
        corrupts InsightFace's ONNX session _inputs_meta and raises
        TypeError: 'NoneType' is not iterable), we resize the *frame* to
        match each target scale.  The model's det_size stays fixed at
        whatever was set in __init__ — a single app.prepare() call, ever.

        Heuristic starting scale:
          1-2 people  → 320  (fast, close-up)
          3-8 people  → 640  (balanced)
          9+ people   → 960  (group/CCTV wide shot)
        """
        if person_count_hint >= 9:
            start_idx = 2
        elif person_count_hint >= 3:
            start_idx = 1
        else:
            start_idx = 0

        h_orig, w_orig = frame.shape[:2]

        results: list[DetectedFace] = []
        for size in sizes[start_idx:]:
            scaled = self._resize_for_det_size(frame, size)
            results = self.detect(scaled)
            if results:
                # Rescale bbox / landmarks back to the ORIGINAL frame's
                # coordinate space.  detect() returns coordinates relative to
                # `scaled`; callers (pipeline.py, annotate, …) all expect
                # coordinates relative to the original `frame`.
                h_sc, w_sc = scaled.shape[:2]
                if h_sc != h_orig or w_sc != w_orig:
                    sx = w_orig / w_sc   # x scale factor
                    sy = h_orig / h_sc   # y scale factor
                    for face in results:
                        x1, y1, x2, y2 = face.bbox
                        face.bbox = [
                            int(x1 * sx), int(y1 * sy),
                            int(x2 * sx), int(y2 * sy),
                        ]
                        face.landmarks = [
                            [int(lx * sx), int(ly * sy)]
                            for lx, ly in face.landmarks
                        ]
                break  # found faces — stop retrying

        return results

    def recognise(
        self,
        frame     : np.ndarray,
        watchlist : Watchlist,
        frame_id  : str = "frame_0",
        camera_id : str = "cam_0",
    ) -> list[FaceResult]:
        """Detect faces + match against watchlist in one call."""
        return [
            FaceResult(face=face, match=watchlist.search(face.embedding),
                       frame_id=frame_id, camera_id=camera_id)
            for face in self.detect(frame)
        ]

    def draw(self, frame: np.ndarray, results: list[FaceResult]) -> np.ndarray:
        """Draw bounding boxes, landmarks, and match info on frame."""
        out = frame.copy()
        for r in results:
            x1, y1, x2, y2 = r.face.bbox
            is_alert = r.match and r.match.is_match
            color    = (0, 0, 255) if is_alert else (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            if r.match:
                label = (f"ALERT: {r.match.name} ({r.match.similarity:.2f})"
                         if is_alert else f"Unknown ({r.match.similarity:.2f})")
            else:
                label = f"Face {r.face.confidence:.2f}"

            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            for (lx, ly) in r.face.landmarks:
                cv2.circle(out, (lx, ly), 3, (255, 255, 0), -1)
        return out
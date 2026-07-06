"""
Run this directly (no FastAPI, no server needed):
    python t2.py sandeep-3.jpeg

Loads the image, runs the SAME face recogniser used by the live pipeline,
gets a real embedding, and calls a CLEAN version of _match directly
(no diagnostic queries interleaved between execute() and fetchone(),
which was the bug in the previous version of this test).
"""
import sys
import cv2
import numpy as np
import psycopg2.extras
from pipeline import SmartCityPipeline


def clean_match(registry, table: str, embedding: np.ndarray, threshold: float):
    """Exact original _match logic, no debug queries in between."""
    if not registry._available:
        print("Registry not available")
        return None, 0.0

    embedding = np.asarray(embedding, dtype=np.float32)

    with registry._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
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

    print(f"Raw row from query: {row}")

    if row is None:
        return None, 0.0

    similarity = float(row["similarity"])
    if similarity >= threshold:
        return {"person_id": row["person_id"], "name": row["name"]}, similarity
    return None, similarity


image_path = sys.argv[1] if len(sys.argv) > 1 else "sandeep-3.jpeg"

print("Loading pipeline (this takes a moment)...")
pipeline = SmartCityPipeline(
    camera_id="cam_01",
    vehicle_model_size="yolov8m",
    face_ctx_id=-1,
    anpr_interval=1,
    inference_max_side=0,
    plate_model_path="weights/license_plate_yolov8n.pt",
    anpr_min_confidence=0.10,
    enhance_gamma=True,
    enhance_clahe=True,
    enhance_zero_dce=True,
    gamma_target_mean=128.0,
    clahe_clip_limit=3.0,
)

frame = cv2.imread(image_path)
if frame is None:
    print(f"ERROR: could not load {image_path}")
    sys.exit(1)

print(f"Loaded {image_path}, shape={frame.shape}")

face_results = pipeline.recogniser.recognise(frame, pipeline.watchlist, "debug", "debug")
print(f"Detected {len(face_results or [])} face(s)")

for i, fr in enumerate(face_results or []):
    emb = fr.face.embedding
    print(f"contiguous={emb.flags['C_CONTIGUOUS']}, strides={emb.strides}, base_is_none={emb.base is None}")
    emb = np.ascontiguousarray(emb, dtype=np.float32)  # force a clean contiguous copy
    print(f"AFTER FIX: contiguous={emb.flags['C_CONTIGUOUS']}")

    person, sim = clean_match(pipeline.person_registry, "face_embeddings", emb, threshold=-1.0)
    print(f"MATCH RESULT: person={person}, similarity={sim}")   
    print(f"\n--- Face {i} ---")
    print(f"embedding type={type(emb)}, shape={getattr(emb, 'shape', None)}, dtype={getattr(emb, 'dtype', None)}")

    person, sim = clean_match(pipeline.person_registry, "face_embeddings", emb, threshold=-1.0)
    print(f"MATCH RESULT: person={person}, similarity={sim}")
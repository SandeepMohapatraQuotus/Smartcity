"""
Smart City Video Analytics — FastAPI Server
--------------------------------------------
Path : py-server/main.py

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Swagger UI (auto-generated):
    http://localhost:8000/docs
"""
import uvicorn
import asyncio
import base64
import io
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import UploadFile, File, Form
from typing import List
import traceback
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError

from pipeline import SmartCityPipeline, FrameEvent
from clashifiers.identity_resolver import IdentityResolver


# ─── Globals ──────────────────────────────────────────────────────────────────

pipeline: Optional[SmartCityPipeline] = None

event_buffer : deque[dict] = deque(maxlen=500)
alert_buffer : deque[dict] = deque(maxlen=200)

# Active WebSocket clients — push updates to all of them
_ws_clients: set[WebSocket] = set()

stream_state = {
    "running":     False,
    "source":      None,
    "camera_id":   None,
    "started_at":  None,
    "frame_count": 0,
    "error":       None,
}

_start_time = time.time()


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all CNN models once at startup — not per request."""
    global pipeline
    print("[Server] Loading pipeline models ...")
    pipeline = SmartCityPipeline(
        camera_id           = "cam_01",
        vehicle_model_size  = "yolov8m",
        face_ctx_id         = -1,     # -1 = CPU;  set 0 if you have an NVIDIA GPU
        anpr_interval       = 1,      # Process ANPR on every sampled frame
        inference_max_side  = 0,      # Disable downscaling to preserve plate details
        # ── ANPR: YOLOv11 plate detector ────────────────────────────
        # Path to the fine-tuned plate-detection weights (see services/anpr.py).
        # Falls back to the contour heuristic automatically if this file is
        # missing, so it's safe to leave set even before you've downloaded it.
        plate_model_path    = "weights/license_plate_yolov8n.pt",
        anpr_min_confidence = 0.10,
        # ── Night enhancement chain: Gamma Correction -> CLAHE -> Zero-DCE++ ──
        # All three stages run in sequence on every frame routed to enhancement.
        # Flip any of these to False to drop that stage from the chain.
        enhance_gamma        = True,
        enhance_clahe         = True,
        enhance_zero_dce      = True,
        gamma_target_mean     = 128.0,
        clahe_clip_limit      = 3.0,
    )
    print("[Server] Ready.\n")
    yield
    if stream_state["running"]:
        stream_state["running"] = False
    print("[Server] Shutdown.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Smart City Video Analytics",
    description = "Day/Night · Vehicle Detection (YOLOv8) · Face Recognition",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class Base64Request(BaseModel):
    image_b64 : str
    camera_id : str            = "cam_01"
    frame_id  : Optional[str] = None

class StreamStartRequest(BaseModel):
    source    : str            # RTSP URL or file path
    camera_id : str = "cam_01"



# Optional: register HEIC/HEIF support with Pillow if the plugin is installed
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def _decode_bytes(raw: bytes) -> np.ndarray:
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file received.")

    # ── Fast path: OpenCV (JPEG, PNG, BMP, TIFF, WebP, etc.) ───────────────
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is not None:
        return frame

    # ── Fallback: Pillow (HEIC/HEIF, AVIF, GIF, ICO, CMYK JPEGs, etc.) ──────
    try:
        img = Image.open(io.BytesIO(raw))
        img.seek(0)  # first frame if animated (GIF/WEBP)
        img = img.convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return frame
    except (UnidentifiedImageError, OSError, Exception):
        raise HTTPException(
            status_code=400,
            detail=(
                "Cannot decode image. Supported formats: JPEG, PNG, BMP, TIFF, "
                "WEBP, GIF, HEIC/HEIF, ICO. The file may be corrupt or truncated."
            ),
        )

def _decode_b64(b64: str) -> np.ndarray:
    try:
        return _decode_bytes(base64.b64decode(b64))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

def _serialise(event: FrameEvent) -> dict:
    """Convert FrameEvent → JSON-safe dict.
    faces is now a list of plain dicts (no nested numpy embeddings) so no
    stripping step is needed here any more.
    """
    return event.to_dict()

def _store(event: FrameEvent):
    d = _serialise(event)
    event_buffer.append(d)
    for alert in event.alerts:
        alert_buffer.append(alert)
    # Broadcast latest state to all connected WebSocket clients
    asyncio.ensure_future(_broadcast_update())


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "Smart City Video Analytics API",
        "docs":    "/docs",
        "status":  "ok",
    }


@app.get("/status", tags=["Health"])
async def status():
    return {
        "status":    "ok",
        "pipeline":  "ready" if pipeline else "loading",
        "uptime_sec": round(time.time() - _start_time, 1),
        "models": {
            "day_night_classifier": "MobileNetV2  (ImageNet pretrained)",
            "vehicle_detector":     "YOLOv8m      (COCO pretrained)",
            "face_recogniser":      "RetinaFace + ArcFace  (InsightFace buffalo_l)",
            "anpr_plate_detector":  "YOLOv11n (fine-tuned)" if (pipeline and pipeline.anpr._yolo_detector is not None) else "contour heuristic (fallback)",
            "night_enhancer":       (
                "Gamma Correction -> CLAHE -> Zero-DCE++ (chained)"
                if pipeline else "loading"
            ),
        },
    }


# ─── Analysis ─────────────────────────────────────────────────────────────────

@app.post("/analyse/frame", tags=["Analysis"])
async def analyse_frame(
    file      : UploadFile = File(..., description="JPEG or PNG image"),
    camera_id : str        = Form(default="cam_01"),
):
    """
    Upload an image file → run all 3 classifiers → return JSON.
    """
    frame = _decode_bytes(await file.read())
    pipeline.camera_id = camera_id
    event = await asyncio.to_thread(pipeline.process_frame, frame)
    _store(event)
    return JSONResponse(content=_serialise(event))



@app.post("/analyse/base64", tags=["Analysis"])
async def analyse_base64(body: Base64Request):
    """
    Send a base64-encoded image (browser canvas, IoT device, mobile).
    Same response shape as /analyse/frame.
    """
    frame = _decode_b64(body.image_b64)
    pipeline.camera_id = body.camera_id
    event = pipeline.process_frame(frame)
    if body.frame_id:
        event.frame_id = body.frame_id
    _store(event)
    return JSONResponse(content=_serialise(event))


@app.post("/analyse/frame/annotated", tags=["Analysis"])
async def analyse_frame_annotated(
    file      : UploadFile = File(...),
    camera_id : str        = Form(default="cam_01"),
):
    """
    Upload an image → returns annotated JPEG with bounding boxes drawn.
    Useful for dashboard live preview.
    """
    frame = _decode_bytes(await file.read())
    pipeline.camera_id = camera_id
    event     = pipeline.process_frame(frame)
    _store(event)
    annotated = pipeline.annotate(frame, event)
    _, buf    = cv2.imencode(".jpg", annotated)
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type = "image/jpeg",
        headers    = {"X-Frame-Id": event.frame_id},
    )


# ─── Individual Classifiers ───────────────────────────────────────────────────

@app.post("/classify/day-night", tags=["Classifiers"])
async def classify_day_night(
    file: UploadFile = File(..., description="JPEG or PNG image"),
):
    """
    Run **only** the Day/Night classifier on an uploaded image.
    """
    frame = _decode_bytes(await file.read())
    result = pipeline.day_night.predict(frame)
    return JSONResponse(content=result)


@app.post("/enhance/frame", tags=["Classifiers"])
async def enhance_frame(
    file: UploadFile = File(..., description="JPEG or PNG image (preferably low-light)"),
):
    """
    Run the full night-enhancement CHAIN on an uploaded image:
        Gamma Correction  →  CLAHE  →  Zero-DCE++
    Each stage feeds the next; any stage disabled at pipeline startup is
    skipped. Returns the enhanced JPEG with headers describing what ran.
    """
    frame    = _decode_bytes(await file.read())
    enhanced = pipeline.enhancer.enhance(frame)
    _, buf   = cv2.imencode(".jpg", enhanced)
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type = "image/jpeg",
        headers    = {
            "X-Enhancement-Method": pipeline.enhancer.method,
            "X-Stages-Applied":     ",".join(pipeline.enhancer.last_stages_applied),
            "X-Gamma-Used":         str(pipeline.enhancer.last_gamma),
        },
    )


@app.post("/detect/vehicles", tags=["Classifiers"])
async def detect_vehicles(
    file      : UploadFile = File(..., description="JPEG or PNG image"),
    camera_id : str        = Form(default="cam_01"),
):
    """
    Run **only** the Vehicle Detector (YOLOv8) on an uploaded image.
    """
    frame  = _decode_bytes(await file.read())
    result = pipeline.vehicles.detect(
        frame,
        frame_id  = f"vdet_{int(time.time()*1000)}",
        camera_id = camera_id,
    )
    return JSONResponse(content=result.to_dict())


@app.post("/detect/persons", tags=["Classifiers"])
async def detect_persons(
    file      : UploadFile = File(..., description="JPEG or PNG image"),
    camera_id : str        = Form(default="cam_01"),
):
    """
    Run **only** the Person Detector (YOLOv8) on an uploaded image.
    """
    frame  = _decode_bytes(await file.read())
    result = pipeline.person_detector.detect(
        frame,
        frame_id  = f"pdet_{int(time.time()*1000)}",
        camera_id = camera_id,
    )
    return JSONResponse(content=result.to_dict())

def _extract_vehicle_boxes(vehicle_result) -> list:
    """
    Pull bbox list out of VehicleDetectionResult regardless of the exact
    attribute name used for the detections list / bbox field.
    """
    for attr in ("vehicles", "detections", "objects", "results", "items"):
        items = getattr(vehicle_result, attr, None)
        if items is not None:
            break
    else:
        raise AttributeError(
            f"VehicleDetectionResult has no known detections attribute. "
            f"Available: {[a for a in dir(vehicle_result) if not a.startswith('_')]}"
        )

    boxes = []
    for v in items:
        for box_attr in ("bbox", "box", "xyxy"):
            box = getattr(v, box_attr, None)
            if box is not None:
                boxes.append(list(box))
                break
    return boxes

@app.post("/anpr/read", tags=["ANPR"])
async def anpr_read(
    file      : UploadFile = File(..., description="JPEG or PNG image containing a vehicle"),
    camera_id : str        = Form(default="cam_01"),
):
    """
    Run **ANPR** (Automatic Number Plate Recognition) on an uploaded image.

    Vehicle detection (YOLOv8) runs first, and plate search is restricted to
    each detected vehicle's bounding box — same behaviour as the live
    pipeline (fewer false positives from background clutter, signs, etc.)

    Two-stage pipeline:
      1. Plate localisation  (YOLOv11 fine-tuned detector, or contour heuristic fallback)
      2. OCR                 (EasyOCR preferred, pytesseract fallback)
    """
    frame = _decode_bytes(await file.read())
    frame_id = f"anpr_{int(time.time()*1000)}"

    vehicle_result = pipeline.vehicles.detect(frame, frame_id=frame_id, camera_id=camera_id)
    vehicle_boxes  = _extract_vehicle_boxes(vehicle_result)


    result = pipeline.anpr.read_plates_in_vehicles(
        frame, vehicle_boxes, frame_id=frame_id, camera_id=camera_id,
    )
    return JSONResponse(content=result.to_dict())

@app.post("/anpr/read/annotated", tags=["ANPR"])
async def anpr_read_annotated(
    file      : UploadFile = File(...),
    camera_id : str        = Form(default="cam_01"),
):
    """
    Upload an image → returns annotated JPEG with vehicle boxes, plate
    bounding boxes, and recognised text drawn directly on the image.
    """
    frame = _decode_bytes(await file.read())
    frame_id = f"anpr_{int(time.time()*1000)}"

    vehicle_result = pipeline.vehicles.detect(frame, frame_id=frame_id, camera_id=camera_id)
    vehicle_boxes  = _extract_vehicle_boxes(vehicle_result)

    result = pipeline.anpr.read_plates_in_vehicles(
        frame, vehicle_boxes, frame_id=frame_id, camera_id=camera_id,
    )

    annotated = pipeline.vehicles.draw(frame, vehicle_result)
    annotated = pipeline.anpr.draw(annotated, result)

    _, buf = cv2.imencode(".jpg", annotated)
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type = "image/jpeg",
        headers    = {
            "X-Vehicle-Count": str(len(vehicle_boxes)),   # ← was vehicle_result.vehicles
            "X-Plate-Count":   str(len(result.plates)),
            "X-OCR-Engine":    result.ocr_engine,
        },
    )

# ─── Dehazing ─────────────────────────────────────────────────────────────────

@app.post("/dehaze/frame", tags=["Dehazing"])
async def dehaze_frame(
    file     : UploadFile = File(..., description="Hazy / foggy JPEG or PNG image"),
    strength : float      = Form(default=0.90, description="Haze removal strength 0.5–1.0 (default 0.90)"),
):
    """
    Remove haze, fog, or smoke from an uploaded image.
    """
    pipeline.dehazer.omega = max(0.3, min(1.0, strength))
    frame  = _decode_bytes(await file.read())
    result = pipeline.dehazer.dehaze(frame)
    _, buf = cv2.imencode(".jpg", result.dehazed_frame)
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type = "image/jpeg",
        headers    = {
            "X-Atm-Light": str(result.atm_light),
            "X-Method":    result.method,
        },
    )


@app.post("/dehaze/frame/compare", tags=["Dehazing"])
async def dehaze_frame_compare(
    file: UploadFile = File(..., description="Hazy / foggy JPEG or PNG image"),
):
    """
    Remove haze from an image and return a **side-by-side comparison** JPEG.
    """
    frame  = _decode_bytes(await file.read())
    result = pipeline.dehazer.dehaze(frame)

    # Draw divider line and labels on both halves
    orig    = frame.copy()
    dehazed = result.dehazed_frame.copy()
    h, w    = orig.shape[:2]

    label_cfg = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2)
    for img, text in [(orig, "HAZY (original)"), (dehazed, "DEHAZED (DCP)")]:
        (tw, th), _ = cv2.getTextSize(text, **label_cfg)
        # Dark background bar
        cv2.rectangle(img, (0, 0), (tw + 14, th + 14), (20, 20, 20), -1)
        cv2.putText(img, text, (7, th + 7), color=(0, 230, 255), **label_cfg)

    # Concatenate side by side
    comparison = np.concatenate([orig, dehazed], axis=1)
    # Draw a thin white divider line in the middle
    cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 2)

    _, buf = cv2.imencode(".jpg", comparison)
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type = "image/jpeg",
        headers    = {
            "X-Atm-Light": str(result.atm_light),
            "X-Method":    result.method,
        },
    )

@app.post("/person/add")
async def add_person(name: str = Form(...), images: List[UploadFile] = File(...)):
    """
    Register a person by name using one or more reference images.
    Images can be clear face photos, full-body/CCTV shots, or a mix.
    For each image: a face embedding is stored if a face is found,
    and a body Re-ID embedding is stored if a person is detected.
    An image only counts as 'skipped' if neither succeeds.
    """
    decoded_images = []
    for upload in images:
        raw = await upload.read()
        frame = _decode_bytes(raw)
        if frame is not None:
            decoded_images.append(frame)
 
    if not decoded_images:
        return {"error": "No valid images could be decoded."}
 
    outcome = pipeline.person_registry.add_person(
        name=name,
        images=decoded_images,
        face_recogniser=pipeline.recogniser,
        person_detector=pipeline.person_detector,
        reidentifier=pipeline.reidentifier,
    )

    if outcome.registry_unavailable:
        raise HTTPException(
            status_code=503,
            detail=(
                "Person registry is unavailable: pgvector extension is not installed "
                "on the PostgreSQL server. "
                "Run: sudo apt install postgresql-17-pgvector "
                "then re-run pgvector_setup.sql and restart the server."
            ),
        )

    status = "ok" if (outcome.face_embeddings_added or outcome.body_embeddings_added) else "warning_no_usable_images"
    return {
        "person_id": outcome.person_id,
        "name": outcome.name,
        "images_received": outcome.images_received,
        "face_embeddings_added": outcome.face_embeddings_added,
        "body_embeddings_added": outcome.body_embeddings_added,
        "images_skipped": outcome.images_skipped,
        "status": status,
        "errors": outcome.errors if status != "ok" else [],
    }
 
 
@app.delete("/person/{person_id}")
async def remove_person(person_id: str):
    removed = pipeline.person_registry.remove(person_id)
    if not removed:
        return {"error": f"No person found with id {person_id}"}
    return {"status": "removed", "person_id": person_id}
 
 
@app.get("/person")
async def list_people():
    return {"people": pipeline.person_registry.list_people()}


# =====================================================================
# CORRECTED DEBUG ROUTE — uses pipeline.recogniser.recognise(...) and
# pipeline.watchlist, matching the real pipeline.py attribute names.
# =====================================================================
@app.get("/person/debug_status")
async def debug_registry_status():
    """
    Reports whether PersonRegistry actually has a live DB connection right
    now. If '_available' is False, every match silently returns null/0.0
    with no visible error — this endpoint exposes that hidden state.
    """
    reg = pipeline.person_registry
    print(f"DEBUG debug_status: id(reg)={id(reg)}, id(reg._conn)={id(reg._conn)}")
    status = {"available": reg._available}
    if reg._available:
        try:
            with reg._conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM persons")
                status["persons_count"] = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM face_embeddings")
                status["face_embeddings_count"] = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM body_embeddings")
                status["body_embeddings_count"] = cur.fetchone()[0]
                cur.execute("SELECT current_database()")
                status["connected_database"] = cur.fetchone()[0]
        except Exception as e:
            status["query_error"] = str(e)
    else:
        status["reason"] = "Connection failed at startup — check server console logs for the PersonRegistry warning printed when SmartCityPipeline() was constructed."
    return status
@app.post("/person/debug_match_all")
async def debug_match_all_faces(image: UploadFile = File(...)):
    """
    Upload a test image (can contain multiple people). Only returns faces
    and bodies where a registered person was actually matched (i.e. passed
    the similarity threshold). Non-matches are silently dropped — bbox is
    only included when there's a real match.
    """
    raw = await image.read()
    frame = _decode_bytes(raw)
    if frame is None:
        return {"error": "Could not decode image."}

    output = {"faces": [], "bodies": []}

    # --- Check every face in the image ---
    try:
        detected_faces = pipeline.recogniser.detect(frame)
        for i, face in enumerate(detected_faces or []):
            if face.embedding is None:
                continue

            raw_person, raw_sim = pipeline.person_registry._match(
                "face_embeddings",
                face.embedding,
                threshold=pipeline.person_registry.face_sim_threshold,
            )

            if raw_person is not None:
                output["faces"].append({
                    "face_index": i,
                    "bbox": list(face.bbox),
                    "detection_confidence": float(face.confidence),
                    "best_match": raw_person,
                    "similarity": raw_sim,
                })
    except Exception as e:
        output["face_error"] = str(e)
        output["face_error_type"] = type(e).__name__
        output["face_error_trace"] = traceback.format_exc()

    # --- Check every detected person's body ---
    try:
        det_result = pipeline.person_detector.detect(frame, frame_id="debug", camera_id="debug")
        for i, det in enumerate(getattr(det_result, "detections", [])):
            x1, y1, x2, y2 = det.bbox
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            vec = pipeline.reidentifier.embed(crop)
            if vec is None:
                continue

            raw_person, raw_sim = pipeline.person_registry._match(
                "body_embeddings",
                vec,
                threshold=pipeline.person_registry.body_sim_threshold,
            )

            # Only keep this body if it actually passed the threshold
            if raw_person is not None:
                output["bodies"].append({
                    "person_index": i,
                    "bbox": list(det.bbox),
                    "best_match": raw_person,
                    "similarity": raw_sim,
                })
    except Exception as e:
        output["body_error"] = str(e)
        output["body_error_type"] = type(e).__name__
        output["body_error_trace"] = traceback.format_exc()

    return output# ─── Stream ───────────────────────────────────────────────────────────────────
async def _stream_worker(source: str, camera_id: str):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        stream_state["running"] = False
        stream_state["error"]   = f"Cannot open: {source}"
        return

    stream_state["frame_count"] = 0
    pipeline.camera_id = camera_id

    PROCESS_EVERY_N = 1  # Process every frame for high-speed scenarios

    try:
        frame_idx = 0
        while stream_state["running"]:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % PROCESS_EVERY_N != 0:
                stream_state["frame_count"] += 1
                continue  # skip inference, just advance the video

            event = await asyncio.get_event_loop().run_in_executor(
                None, pipeline.process_frame, frame
            )
            _store(event)
            stream_state["frame_count"] += 1
            await asyncio.sleep(0)
    finally:
        cap.release()
        stream_state["running"] = False
        print(f"[Stream] Stopped — {stream_state['frame_count']} frames processed.")

@app.post("/stream/start", tags=["Stream"])
async def stream_start(body: StreamStartRequest, background_tasks: BackgroundTasks):
    """Start processing an RTSP stream as a background task."""
    if stream_state["running"]:
        raise HTTPException(409, "A stream is already running. POST /stream/stop first.")

    stream_state.update({
        "running":     True,
        "source":      body.source,
        "camera_id":   body.camera_id,
        "started_at":  time.time(),
        "frame_count": 0,
        "error":       None,
    })
    background_tasks.add_task(_stream_worker, body.source, body.camera_id)
    return {"message": "Stream started.", "source": body.source, "camera_id": body.camera_id}


@app.post("/stream/stop", tags=["Stream"])
async def stream_stop():
    """Signal the running stream to stop."""
    if not stream_state["running"]:
        raise HTTPException(400, "No stream is currently running.")
    stream_state["running"] = False
    return {"message": "Stop signal sent.", "frames_processed": stream_state["frame_count"]}


@app.get("/stream/status", tags=["Stream"])
async def stream_status():
    return {
        "running":          stream_state["running"],
        "source":           stream_state["source"],
        "camera_id":        stream_state["camera_id"],
        "started_at":       stream_state["started_at"],
        "frames_processed": stream_state["frame_count"],
        "error":            stream_state["error"],
    }


# ─── MJPEG Live Stream ────────────────────────────────────────────────────────

def _draw_all(frame: np.ndarray, event: "FrameEvent") -> np.ndarray:
    """
    Draws all classifier results onto the frame and returns the annotated copy.
    """
    annotated = frame.copy()
    if event.vehicles:
        annotated = pipeline.vehicles.draw(annotated, event.vehicles)
    if event.persons:
        annotated = pipeline.person_detector.draw(annotated, event.persons)

    # ── Overlay identified person names ──────────────────────────────
    if event.identified_people and event.persons:
        id_by_track = {p["track_id"]: p for p in event.identified_people}
        for det in event.persons.detections:
            identity = id_by_track.get(det.track_id)
            if not identity:
                continue
            x1, y1 = det.bbox[0], det.bbox[1]
            method  = identity.get("method", "")
            name    = identity["name"]
            sim     = identity["similarity"]
            label   = f"{name}  {sim:.2f}  [{method}]"
            # green for face match, cyan for body match
            color = (0, 230, 80) if method == "face" else (0, 220, 255)
            # dark backing bar so text is readable on any background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 14), (x1 + tw + 8, y1 - 2), (20, 20, 20), -1)
            cv2.putText(annotated, label, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ── Face boxes (now plain dicts, not FaceResult objects) ─────────
    for f in event.faces:
        x1, y1, x2, y2 = f["bbox"]
        is_alert = f["best_match"] is not None
        color = (0, 0, 255) if is_alert else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = (f"{f['best_match']['name']} ({f['similarity']:.2f})"
                 if is_alert else f"Unknown ({f['confidence']:.2f})")
        cv2.putText(annotated, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return annotated



def _open_capture(source: str) -> cv2.VideoCapture:
    """
    Opens a VideoCapture and raises a clean 400 if it fails.
    """
    target = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(target)

    if not cap.isOpened():
        cap.release()
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cannot open source {source!r}. "
                "No webcam found (device index out of range), "
                "RTSP unreachable, or file path invalid. "
                "Try: an RTSP URL, a local video file path, or GET /stream/sources."
            ),
        )
    return cap


async def _mjpeg_generator(cap: cv2.VideoCapture, jpeg_quality: int = 75):
    """Yields MJPEG frames. cap is already validated open."""
    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                await asyncio.sleep(0.05)
                continue

            event = await asyncio.to_thread(pipeline.process_frame, frame)
            _store(event)

            annotated = _draw_all(frame, event)
            dn_label  = f"{event.day_night['label'].upper()}  {event.day_night['confidence']:.2f}"
            if event.enhanced:
                dn_label += "  [ENHANCED]"
            cv2.putText(
                annotated, dn_label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
            )

            ok, buf = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
            await asyncio.sleep(0)

    finally:
        cap.release()


@app.get(
    "/stream/mjpeg",
    tags=["Stream"],
    summary="Live MJPEG stream",
    description=(
        "Returns a multipart/x-mixed-replace stream. "
        "Paste the URL directly into a browser tab or an <img> src attribute. "
        "Pass ?source= to override the stream URL (default: webcam 0). "
        "Pass ?quality= (1-95) to trade bandwidth for image quality."
    ),
)
async def stream_mjpeg(source: str = None, quality: int = 75):
    """
    GET /stream/mjpeg
    """
    resolved = source or stream_state.get("source")
    if not resolved:
        raise HTTPException(
            status_code=400,
            detail=(
                "No source specified and no stream is currently running. "
                "Either pass ?source= or POST /stream/start first."
            ),
        )

    cap = _open_capture(resolved)

    return StreamingResponse(
        _mjpeg_generator(cap, jpeg_quality=quality),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma":        "no-cache",
            "Expires":       "0",
        },
    )


@app.get("/stream/sources", tags=["Stream"], summary="List available camera devices")
def stream_sources():
    """
    Probes device indices 0-9 and returns which ones OpenCV can open.
    """
    found = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found.append({
                "index":  i,
                "source": str(i),
                "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps":    cap.get(cv2.CAP_PROP_FPS),
            })
            cap.release()

    return {
        "cameras_found": len(found),
        "devices":       found,
        "hint": (
            "No cameras found - use an RTSP URL or a local video file path instead."
            if not found else
            "Pass ?source=<index> to /stream/mjpeg"
        ),
    }



# ─── Watchlist ────────────────────────────────────────────────────────────────

@app.post("/watchlist/add", tags=["Watchlist"])
async def watchlist_add(
    name      : str                 = Form(..., description="Display name  e.g. 'Sandeep'"),
    person_id : Optional[str]       = Form(None, description="Existing person_id to add more photos to (optional)"),
    files     : List[UploadFile]    = File(..., description="One or more reference face / body photos (JPEG / PNG)"),
):
    """
    Add or update a person on the **face watchlist AND person registry**.

    Accepts **multiple reference photos in a single call** — this is important:
    if the frontend uploads N photos by calling this endpoint N times (one
    photo each), and doesn’t carry the returned `person_id` across calls, each
    call previously minted a fresh UUID, producing N duplicate personas.

    Fix: upload all photos in **one** multipart call.  Or, on subsequent calls,
    pass back the `person_id` returned by the first call.

    Deduplication: if no `person_id` is supplied and a person with the same
    `name` (case-insensitive) already exists in the registry, the new photos
    are merged into that existing entry instead of creating a new one.
    """
    decoded_images: list[np.ndarray] = []
    for upload in files:
        raw = await upload.read()
        try:
            frame = _decode_bytes(raw)
            if frame is not None:
                decoded_images.append(frame)
        except Exception:
            pass  # individual bad uploads are skipped; errors surface in outcome.errors

    if not decoded_images:
        raise HTTPException(status_code=422, detail="No valid images could be decoded from the uploaded files.")

    # ── Register in the full PersonRegistry (pgvector) ───────────────────────
    outcome = pipeline.person_registry.add_person(
        name=name,
        images=decoded_images,
        face_recogniser=pipeline.recogniser,
        person_detector=pipeline.person_detector,
        reidentifier=pipeline.reidentifier,
        person_id=person_id,
    )

    if outcome.registry_unavailable:
        raise HTTPException(
            status_code=503,
            detail=(
                "Person registry unavailable: pgvector extension not installed. "
                "Run: sudo apt install postgresql-17-pgvector "
                "then re-run pgvector_setup.sql and restart the server."
            ),
        )

    # ── pgvector PersonRegistry is the only store now (Watchlist removed) ───
    # The watchlist.add_person() call that used to live here has been removed.

    return {
        "person_id":              outcome.person_id,
        "name":                   outcome.name,
        "images_received":        outcome.images_received,
        "face_embeddings_added":  outcome.face_embeddings_added,
        "body_embeddings_added":  outcome.body_embeddings_added,
        "images_skipped":         outcome.images_skipped,
        "reused_existing_person": outcome.reused_existing_person,
        "note": (
            "Matched an existing registered person by name — merged into that "
            "person_id instead of creating a new one."
            if outcome.reused_existing_person else
            "Created a new person entry."
        ),
        "errors": outcome.errors if outcome.images_skipped else [],
    }


@app.post("/analyse/identify", tags=["Analysis"])
async def analyse_identify(
    file      : UploadFile = File(..., description="JPEG or PNG image"),
    camera_id : str        = Form(default="cam_01"),
):
    """
    Upload an image → run **IdentityResolver** → return per-person identity JSON.

    Differences from `/analyse/frame`:
      - Adaptive face detection: tries 320→640→960 detection grids until faces are found.
      - Spatial face↔body binding: a body box inherits its face’s identity;
        no separate body-only Re-ID vote that often misidentifies people.
      - Body-only matching is OFF by default (cleaner results in CCTV conditions).

    Response shape:
        {
          "frame_id": str,
          "camera_id": str,
          "person_count": int,
          "people": [
            {
              "track_id": int|null,
              "body_bbox": [x1,y1,x2,y2],
              "face_bbox": [x1,y1,x2,y2]|null,
              "person_id": str|null,
              "name": str|null,
              "similarity": float|null,
              "method": "face"|"body"|null
            }, ...
          ],
          "unbound_faces": [ ... ]   # faces with no body bbox match
        }
    """
    frame = _decode_bytes(await file.read())
    import time as _time
    frame_id = f"ident_{int(_time.time()*1000)}"

    resolver = IdentityResolver(
        registry          = pipeline.person_registry,
        face_recogniser   = pipeline.recogniser,
        person_detector   = pipeline.person_detector,
        reidentifier      = pipeline.reidentifier,
        enable_body_matching = False,  # keep OFF until validated separately
    )
    result = await asyncio.to_thread(
        resolver.identify_frame, frame, frame_id, camera_id
    )
    return JSONResponse(content=result)


@app.delete("/watchlist/{person_id}", tags=["Watchlist"])
async def watchlist_remove(person_id: str):
    """Remove a person from the registry by their person_id."""
    removed = pipeline.person_registry.remove(person_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"'{person_id}' not found.")
    return {"message": f"Removed '{person_id}'."}


@app.get("/watchlist", tags=["Watchlist"])
async def watchlist_list():
    """List all registered persons (now backed by pgvector, not in-memory Watchlist)."""
    people = pipeline.person_registry.list_people()
    return {"total": len(people), "people": people}


# ─── WebSocket ───────────────────────────────────────────────────────────────

async def _broadcast_update():
    """
    Build the current snapshot and push it to every connected WS client.
    """
    if not _ws_clients:
        return
    snapshot = _ws_snapshot()
    dead: set[WebSocket] = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_json(snapshot)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def _ws_snapshot() -> dict:
    """Assemble the current state payload sent over WebSocket."""
    latest_event = event_buffer[-1] if event_buffer else None
    alerts = list(alert_buffer)
    return {
        "stream_status": {
            "running":          stream_state["running"],
            "source":           stream_state["source"],
            "camera_id":        stream_state["camera_id"],
            "started_at":       stream_state["started_at"],
            "frames_processed": stream_state["frame_count"],
            "error":            stream_state["error"],
        },
        "latest_event": latest_event,
        "alerts":       alerts,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint — ws://localhost:8000/ws
    """
    await ws.accept()
    _ws_clients.add(ws)
    # Send the current state immediately on connect
    try:
        await ws.send_json(_ws_snapshot())
    except Exception:
        _ws_clients.discard(ws)
        return

    try:
        while True:
            await asyncio.sleep(0.5)
            try:
                await ws.send_json(_ws_snapshot())
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


# ─── Events & Alerts ──────────────────────────────────────────────────────────

@app.get("/events", tags=["Events"])
async def get_events(limit: int = 50):
    """Return the last N frame events from the in-memory buffer."""
    events = list(event_buffer)[-limit:]
    return {"total": len(events), "events": events}


@app.get("/alerts", tags=["Events"])
async def get_alerts(limit: int = 100):
    """Return all face-watchlist alerts triggered so far."""
    alerts = list(alert_buffer)[-limit:]
    return {"total": len(alerts), "alerts": alerts}


@app.delete("/events", tags=["Events"])
async def clear_buffers():
    """Clear both event and alert buffers."""
    event_buffer.clear()
    alert_buffer.clear()
    return {"message": "Buffers cleared."}


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
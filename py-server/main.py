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
import io
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import UploadFile, File, Form
from typing import List
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

# Latest annotated JPEG bytes — written by _stream_worker, read by /stream/mjpeg
# Python's GIL makes a single bytes assignment atomic, so no explicit lock needed.
_latest_annotated_frame: Optional[bytes] = None

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

# ─── Stream ───────────────────────────────────────────────────────────────────
async def _stream_worker(source: str, camera_id: str):
    global _latest_annotated_frame

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
                # For video files: loop back to the beginning instead of stopping
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
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

            # ── Cache latest annotated JPEG for /stream/mjpeg ─────────────
            annotated = await asyncio.get_event_loop().run_in_executor(
                None, pipeline.annotate, frame, event
            )
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            _latest_annotated_frame = buf.tobytes()

            await asyncio.sleep(0)  # yield to event loop
    finally:
        cap.release()
        stream_state["running"] = False
        _latest_annotated_frame = None
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


async def _mjpeg_generator():
    """
    Async generator that yields multipart MJPEG frames.
    Reads from _latest_annotated_frame every ~33 ms (~30 fps cap).
    Sends a placeholder frame when the stream is not yet running.
    """
    boundary = b"--frame"
    while True:
        frame_bytes = _latest_annotated_frame
        if frame_bytes is None:
            # Send a small black placeholder so the <img> doesn't error out
            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder, "Waiting for stream...",
                (120, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2,
            )
            _, buf = cv2.imencode(".jpg", placeholder)
            frame_bytes = buf.tobytes()

        yield (
            boundary + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame_bytes)).encode() + b"\r\n"
            b"\r\n" + frame_bytes + b"\r\n"
        )
        await asyncio.sleep(0.033)  # ~30 fps


@app.get("/stream/mjpeg", tags=["Stream"])
async def stream_mjpeg(source: Optional[str] = None):
    """
    MJPEG HTTP stream — consumed by the MjpegPlayer <img> tag.
    Returns a continuous multipart/x-mixed-replace stream of annotated JPEG frames.
    The `source` query parameter is accepted but ignored here; the active
    stream source is set via POST /stream/start.
    """
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/stream/frame", tags=["Stream"])
async def stream_single_frame():
    """
    Return the **latest single annotated JPEG** frame.
    Useful for a polling-based display or debugging.
    """
    frame_bytes = _latest_annotated_frame
    if frame_bytes is None:
        raise HTTPException(status_code=503, detail="No active stream. POST /stream/start first.")
    return StreamingResponse(
        io.BytesIO(frame_bytes),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"},
    )






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
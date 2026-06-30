"""
Smart City Video Analytics — FastAPI Server
--------------------------------------------
Path : py-server/main.py

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Swagger UI (auto-generated):
    http://localhost:8000/docs

Routes
------
GET    /                          Health check
GET    /status                    Pipeline info + uptime
POST   /analyse/frame             Upload image  → JSON detections (full pipeline)
POST   /analyse/base64            Base64 image  → JSON detections (full pipeline)
POST   /analyse/frame/annotated   Upload image  → annotated JPEG (boxes drawn)
POST   /classify/day-night        Upload image  → day/night label + confidence only
POST   /enhance/frame             Upload image  → enhanced JPEG (Zero-DCE++ / CLAHE)
POST   /detect/vehicles           Upload image  → vehicle detections only (YOLOv8)
POST   /detect/persons            Upload image  → person detections only (YOLOv8)
POST   /detect/persons/annotated  Upload image  → annotated JPEG with person boxes
POST   /anpr/read                 Upload image  → number plate texts (EasyOCR)
POST   /anpr/read/annotated       Upload image  → annotated JPEG with plate labels
POST   /dehaze/frame              Upload hazy image → dehazed JPEG (Dark Channel Prior)
POST   /dehaze/frame/compare      Upload hazy image → side-by-side comparison JPEG
POST   /stream/start              Start RTSP stream (background task)
POST   /stream/stop               Stop running stream
GET    /stream/status             Stream state + frame count
GET    /stream/mjpeg              Live annotated MJPEG stream (paste URL in browser)
POST   /watchlist/add             Upload reference photo → add to watchlist
DELETE /watchlist/{person_id}     Remove person from watchlist
GET    /watchlist                 List all watchlist persons
GET    /events?limit=50           Last N frame events (in-memory)
GET    /alerts?limit=100          All face-match alerts
DELETE /events                    Clear event + alert buffers
"""
import uvicorn
import asyncio
import base64
import io
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from pipeline import SmartCityPipeline, FrameEvent


# ─── Globals ──────────────────────────────────────────────────────────────────

pipeline: Optional[SmartCityPipeline] = None

event_buffer : deque[dict] = deque(maxlen=500)
alert_buffer : deque[dict] = deque(maxlen=200)

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
        camera_id          = "cam_01",
        vehicle_model_size = "yolov8m",
        face_ctx_id        = -1,     # -1 = CPU;  set 0 if you have an NVIDIA GPU
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _decode_bytes(raw: bytes) -> np.ndarray:
    arr   = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Send JPEG or PNG.")
    return frame

def _decode_b64(b64: str) -> np.ndarray:
    try:
        return _decode_bytes(base64.b64decode(b64))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

def _serialise(event: FrameEvent) -> dict:
    """Convert FrameEvent → JSON-safe dict (strips numpy embeddings)."""
    d = event.to_dict()
    for face_rec in d.get("faces", []):
        face_rec.get("face", {}).pop("embedding", None)
    return d

def _store(event: FrameEvent):
    d = _serialise(event)
    event_buffer.append(d)
    for alert in event.alerts:
        alert_buffer.append(alert)


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

    Response includes:
      - day_night  : label + confidence + enhancement flag
      - vehicles   : list of detections (label, bbox, confidence, track_id)
      - faces      : list of detections + watchlist matches
      - alerts     : face-watchlist hits
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

    Response:
      - label              : \"day\" | \"night\"
      - confidence         : float  (0.0 – 1.0)
      - route_to_enhancement : bool — True if frame should be enhanced
    """
    frame = _decode_bytes(await file.read())
    result = pipeline.day_night.predict(frame)
    return JSONResponse(content=result)


@app.post("/enhance/frame", tags=["Classifiers"])
async def enhance_frame(
    file: UploadFile = File(..., description="JPEG or PNG image (preferably low-light)"),
):
    """
    Run **only** the Zero-DCE++ image enhancer on an uploaded image.

    Returns the enhanced image as a JPEG.
    Response headers:
      - X-Enhancement-Method : \"zero_dce++\" | \"clahe\"  (fallback)
    """
    frame    = _decode_bytes(await file.read())
    enhanced = pipeline.enhancer.enhance(frame)
    _, buf   = cv2.imencode(".jpg", enhanced)
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type = "image/jpeg",
        headers    = {"X-Enhancement-Method": pipeline.enhancer.method},
    )


@app.post("/detect/vehicles", tags=["Classifiers"])
async def detect_vehicles(
    file      : UploadFile = File(..., description="JPEG or PNG image"),
    camera_id : str        = Form(default="cam_01"),
):
    """
    Run **only** the Vehicle Detector (YOLOv8) on an uploaded image.

    Response includes:
      - frame_id   : auto-generated
      - camera_id  : echoed back
      - vehicles   : list of { label, bbox, confidence, track_id }
      - count      : total detections
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

    Response includes:
      - person_count : total persons detected
      - detections   : list of { bbox, confidence, track_id, center, area }
    """
    frame  = _decode_bytes(await file.read())
    result = pipeline.person_detector.detect(
        frame,
        frame_id  = f"pdet_{int(time.time()*1000)}",
        camera_id = camera_id,
    )
    return JSONResponse(content=result.to_dict())



@app.post("/anpr/read", tags=["ANPR"])
async def anpr_read(
    file      : UploadFile = File(..., description="JPEG or PNG image containing a vehicle"),
    camera_id : str        = Form(default="cam_01"),
):
    """
    Run **ANPR** (Automatic Number Plate Recognition) on an uploaded image.

    Two-stage pipeline:
      1. Plate localisation  (contour heuristic, or YOLO if weights provided)
      2. OCR                 (EasyOCR preferred, pytesseract fallback)

    Response includes:
      - plate_count  : number of plates found
      - ocr_engine   : which OCR backend was used
      - plates       : list of { bbox, raw_text, cleaned_text, confidence }
    """
    frame  = _decode_bytes(await file.read())
    result = pipeline.anpr.read_plates(
        frame,
        frame_id  = f"anpr_{int(time.time()*1000)}",
        camera_id = camera_id,
    )
    return JSONResponse(content=result.to_dict())


@app.post("/anpr/read/annotated", tags=["ANPR"])
async def anpr_read_annotated(
    file      : UploadFile = File(..., description="JPEG or PNG image containing a vehicle"),
    camera_id : str        = Form(default="cam_01"),
):
    """
    Upload an image → returns annotated JPEG with plate bounding boxes and
    recognised text drawn directly on the image.
    """
    frame  = _decode_bytes(await file.read())
    result = pipeline.anpr.read_plates(
        frame,
        frame_id  = f"anpr_{int(time.time()*1000)}",
        camera_id = camera_id,
    )
    annotated = pipeline.anpr.draw(frame, result)
    _, buf    = cv2.imencode(".jpg", annotated)
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type = "image/jpeg",
        headers    = {
            "X-Plate-Count": str(len(result.plates)),
            "X-OCR-Engine":  result.ocr_engine,
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

    **Algorithm auto-selection:**
    - Standard outdoor haze/fog → **Sky-Aware DCP** with colour correction + sharpening
    - Bright / overcast / sky-dominant → **MSRCR** (Multi-Scale Retinex)

    `strength` controls how aggressively haze is removed (0.5 = gentle, 1.0 = maximum).

    Returns the dehazed image as a JPEG.
    Response headers:
      - X-Atm-Light : estimated fog density
      - X-Method    : `dcp` | `msrcr`
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
    Remove haze from an image and return a **side-by-side comparison** JPEG:
    left half = original hazy image, right half = dehazed output.

    Useful for visual validation on the dashboard.
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
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        stream_state["running"] = False
        stream_state["error"]   = f"Cannot open: {source}"
        return

    stream_state["frame_count"] = 0
    pipeline.camera_id = camera_id

    PROCESS_EVERY_N = 1000  # only run inference every 3rd frame, tune as needed

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
    Calls each classifier's own .draw() method so annotations stay consistent
    with what you see from /analyse/frame/annotated.
    """
    annotated = frame.copy()
    if event.vehicles:
        annotated = pipeline.vehicles.draw(annotated, event.vehicles)
    if event.persons:
        annotated = pipeline.person_detector.draw(annotated, event.persons)
    if event.plates:
        annotated = pipeline.anpr.draw(annotated, event.plates)
    if event.faces:
        annotated = pipeline.recogniser.draw(annotated, event.faces)
    return annotated


def _open_capture(source: str) -> cv2.VideoCapture:
    """
    Opens a VideoCapture and raises a clean 400 if it fails.
    Accepts: "0","1",… for device index  or  any URL / file path.
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
    GET /stream/mjpeg                                  ← reuses POST /stream/start source
    GET /stream/mjpeg?source=/path/to/video.mp4
    GET /stream/mjpeg?source=rtsp://192.168.1.10:554/ch0
    GET /stream/mjpeg?quality=90
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
    Use this to find the right index before calling /stream/mjpeg.
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
    person_id : str        = Form(..., description="Unique person ID  e.g. P001"),
    name      : str        = Form(..., description="Display name"),
    photo     : UploadFile = File(..., description="Reference face photo (JPEG / PNG)"),
):
    """
    Add a person to the face watchlist.
    Uploads a reference photo, extracts ArcFace embedding, stores in memory.
    """
    frame = _decode_bytes(await photo.read())
    try:
        pipeline.watchlist.add_from_photo(person_id, name, frame, pipeline.recogniser)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {
        "message":   f"'{name}' added to watchlist.",
        "person_id": person_id,
        "total":     len(pipeline.watchlist),
    }


@app.delete("/watchlist/{person_id}", tags=["Watchlist"])
async def watchlist_remove(person_id: str):
    """Remove a person from the watchlist by their person_id."""
    removed = pipeline.watchlist.remove(person_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"'{person_id}' not found.")
    return {"message": f"Removed '{person_id}'.", "remaining": len(pipeline.watchlist)}


@app.get("/watchlist", tags=["Watchlist"])
async def watchlist_list():
    """List all persons currently on the watchlist."""
    people = pipeline.watchlist.list_people()
    return {"total": len(people), "people": people}


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
# Smart City Platform — py-server Documentation

> **Last updated:** 2026-07-20 | **Server:** FastAPI + Uvicorn | **Python:** 3.12

---

## 1. What This Server Does

The py-server is the AI/ML backend of the Smart City Platform. It processes images and video streams from cameras, running multiple computer-vision pipelines per frame and returning structured JSON or annotated images over REST and WebSocket.

### Core Capabilities

| Capability | Models Used |
|---|---|
| Day/Night classification | MobileNetV2 (ImageNet pretrained) |
| Night image enhancement | Gamma Correction → CLAHE → Zero-DCE++ (chained) |
| Vehicle detection + tracking | YOLOv8m + BotSORT |
| Person detection + tracking | YOLOv8m + BotSORT |
| Face detection + recognition | RetinaFace + ArcFace (InsightFace buffalo_l) |
| Person identity matching | pgvector (face embeddings + body Re-ID) |
| Number plate reading (ANPR) | YOLOv11n (plate detector) + EasyOCR |
| Haze/fog removal | Dark Channel Prior + MSRCR |
| Live stream processing | RTSP/file via OpenCV background worker |
| Real-time WebSocket push | ws://localhost:8000/ws |

---

## 2. Directory Tree

```
py-server/
├── main.py                              ← FastAPI app — all REST + WebSocket endpoints
├── pipeline.py                          ← SmartCityPipeline — orchestrates all models per frame
├── pg_vector.py                         ← PersonRegistry — pgvector-backed face+body identity store
├── pgvector_setup.sql                   ← SQL to create persons/face_embeddings/body_embeddings tables
├── requirements.txt                     ← pip dependencies
├── yolov8m.pt                           ← YOLOv8m weights (COCO, shared by vehicle+person detectors)
├── yolov8n.pt                           ← YOLOv8n weights (lighter variant)
├── structure.md                         ← This documentation file
├── run.txt                              ← Quick-start run commands
├── .env                                 ← Environment variables (DB DSN etc.)
├── .venv/                               ← Python virtual environment
├── weights/
│   └── license_plate_yolov8n.pt         ← Fine-tuned plate detector (YOLOv11n)
│
├── clashifiers/                         ← All classifier/detector modules
│   ├── __init__.py
│   ├── identity_resolver.py             ← IdentityResolver — unified face+body identity per frame
│   ├── day_night/
│   │   └── main.py                      ← DayNightClassifier (MobileNetV2, two-stage heuristic+CNN)
│   ├── face_recognization/
│   │   └── main.py                      ← FaceRecogniser (RetinaFace+ArcFace), Watchlist, CLAHE preprocess
│   ├── person_detector/
│   │   └── main.py                      ← PersonDetector (YOLOv8m, BotSORT, adaptive_detect)
│   ├── person_reid/
│   │   └── main.py                      ← PersonReIdentifier (OSNet, body embedding extraction)
│   ├── person_registry/
│   │   └── main.py                      ← (legacy stub, replaced by pg_vector.PersonRegistry)
│   └── vechile_detector/
│       └── main.py                      ← VehicleDetector (YOLOv8m, BotSORT tracking)
│
├── services/
│   ├── __init__.py
│   ├── classical_enhance.py             ← Gamma Correction + CLAHE (CPU-only, model-free)
│   ├── zero_dce.py                      ← ZeroDCEEnhancer — chains Gamma→CLAHE→Zero-DCE++
│   ├── anpr.py                          ← ANPRService — plate localisation + EasyOCR
│   └── dehazing.py                      ← DehazingService — DCP + MSRCR
│
└── Zero-DCE_extension/
    └── Zero-DCE++/
        ├── model.py                     ← enhance_net_nopool CNN architecture
        ├── Myloss.py                    ← L_color, L_spa, L_exp, L_TV loss functions
        ├── dataloader.py                ← lowlight_loader PyTorch dataset
        ├── lowlight_train.py            ← Training script (CUDA, 100 epochs)
        ├── lowlight_test.py             ← Inference script
        └── snapshots_Zero_DCE++/
            └── Epoch99.pth              ← Pretrained weights used by ZeroDCEEnhancer
```

---

## 3. API Routes

**Base URL:** http://localhost:8000  |  **Swagger:** http://localhost:8000/docs  |  **WebSocket:** ws://localhost:8000/ws

### Health

| Method | Route | Description |
|---|---|---|
| GET | /status | Pipeline ready state, uptime, model names |

### Analysis — Full Pipeline

| Method | Route | Description |
|---|---|---|
| POST | /analyse/frame | Upload image → run ALL classifiers → return JSON FrameEvent |
| POST | /analyse/frame/annotated | Upload image → return annotated JPEG with all boxes drawn |
| POST | /analyse/identify | Upload image → run IdentityResolver → per-person identity JSON |

### Individual Classifiers

| Method | Route | Description |
|---|---|---|
| POST | /classify/day-night | Day/Night label + confidence + enhancement flag |
| POST | /enhance/frame | Gamma→CLAHE→Zero-DCE++ enhanced JPEG |
| POST | /detect/vehicles | YOLOv8 vehicle detections + track IDs |
| POST | /detect/persons | YOLOv8 person detections + track IDs |

### ANPR

| Method | Route | Description |
|---|---|---|
| POST | /anpr/read | Plate detection + OCR → JSON |
| POST | /anpr/read/annotated | Plate detection + OCR → annotated JPEG |

### Dehazing

| Method | Route | Description |
|---|---|---|
| POST | /dehaze/frame | Dehazed JPEG (strength param 0.5–1.0) |
| POST | /dehaze/frame/compare | Side-by-side comparison JPEG |

### Stream (RTSP / File)

| Method | Route | Description |
|---|---|---|
| POST | /stream/start | Start background stream worker (body: source, camera_id) |
| POST | /stream/stop | Stop the running stream |
| GET | /stream/status | running, source, camera_id, frames_processed, error |
| GET | /stream/mjpeg | MJPEG HTTP stream — consumed by img tag (~30fps) |
| GET | /stream/frame | Latest single annotated JPEG (for polling) |

### Watchlist / Person Registry

| Method | Route | Description |
|---|---|---|
| POST | /watchlist/add | Upload reference photo(s) → extract face+body embeddings → store in pgvector |
| DELETE | /watchlist/{person_id} | Remove person from registry |
| GET | /watchlist | List all registered persons with face+body ref counts |

### Events & Alerts

| Method | Route | Description |
|---|---|---|
| GET | /events?limit=50 | Last N FrameEvents from ring buffer |
| GET | /alerts?limit=100 | All face/person registry hit alerts |
| DELETE | /events | Clear both buffers |

### WebSocket

| Endpoint | Description |
|---|---|
| ws://localhost:8000/ws | Persistent connection — server pushes snapshot on every processed frame + every 500ms heartbeat |

WebSocket payload shape:
```json
{
  stream_status: { running: true, source: ..., camera_id: cam_01, frames_processed: 42, error: null },
  latest_event:  { ...full FrameEvent dict... },
  alerts:        [ ...all accumulated alerts... ]
}
```

---

## 4. Pipeline Flow (process_frame)

Every frame goes through these stages in order inside SmartCityPipeline.process_frame():

```
BGR Frame
  │
  ▼
[1] DayNightClassifier.predict()
    BRIGHT_THRESH=160 → day (skip enhancement)
    DARK_THRESH=85    → night → enhancement
    61-159 brightness → MobileNetV2 CNN decides
  │
  ├── if night ──▶ [2] ZeroDCEEnhancer.enhance()
  │                     Gamma Correction (adaptive, target mean=128)
  │                     → CLAHE (clip=3.0, tile=8x8, LAB L-channel)
  │                     → Zero-DCE++ (CNN, Epoch99.pth)
  │
  ▼ (working frame = enhanced or original)
  │
  ├──[parallel]──▶ [3] VehicleDetector.detect()   ← runs in thread pool
  │
  ├──[sync]──────▶ [4] PersonDetector.adaptive_detect()
  │                     0-3 people  → native resolution
  │                     4-8 people  → upscale to 1280px wide
  │                     9+ people   → upscale to 1920px wide
  │                     min_height filter=40px (kills false positives)
  │                     BotSORT tracker (replaces ByteTrack)
  │                     conf=0.30, iou=0.50
  │
  ├──[sync]──────▶ [5] FaceRecogniser.adaptive_detect()
  │                     _preprocess_for_face_detection() ← CLAHE always-on
  │                     det_thresh=0.35 (was 0.5)
  │                     320px grid for 1-2 people
  │                     640px grid for 3-8 people
  │                     960px grid for 9+ people (retry up if no faces found)
  │                     ArcFace 512-d embedding per face
  │
  ├──[wait]──────▶ [6] VehicleDetector result collected
  │
  ├──────────────▶ [7] Face identity matching via PersonRegistry.match_face()
  │                     pgvector cosine similarity search
  │                     face_sim_threshold=0.70
  │                     → face_watchlist_hit alert if matched
  │
  ├──────────────▶ [8] Per-person body+face binding + identity
  │                     For each detected person body:
  │                       - Find face whose centre is inside body bbox
  │                       - If face found → use face embedding for identity
  │                       - If no face → run OSNet body Re-ID embedding
  │                     PersonRegistry.identify() → face-first, body fallback
  │                     body_sim_threshold=0.82, margin_check=0.06
  │                     → person_registry_hit alert if matched
  │
  └──────────────▶ [9] ANPRService.read_plates_in_vehicles()
                        (runs every anpr_interval frames, default=1)
                        Stage 1: YOLOv11n plate detector (or contour fallback)
                        Stage 2: EasyOCR (or pytesseract fallback)
                        Restricted to vehicle bboxes only

  ▼
FrameEvent (stored in event_buffer, broadcast to all WebSocket clients)
  ▼
pipeline.annotate() → JPEG → _latest_annotated_frame → /stream/mjpeg
```

---

## 5. File-by-File Reference

### main.py — FastAPI Server

Entry point. Loads SmartCityPipeline once at startup via lifespan(). Routes all HTTP and WS requests.

Key globals:
-  — singleton SmartCityPipeline instance
-  — deque(maxlen=500) of FrameEvent dicts
-  — deque(maxlen=200) of alert dicts
-  — set of active WebSocket connections
-  — dict tracking running/source/camera_id/frame_count/error
-  — bytes of last annotated JPEG (written by stream worker, read by /stream/mjpeg)

Key functions:
-  — async background loop: reads frames from cv2.VideoCapture, calls process_frame, annotates, caches JPEG
-  — async generator yielding multipart MJPEG chunks at ~30fps
-  — pushes _ws_snapshot() to all connected WS clients
-  — builds {stream_status, latest_event, alerts} dict
-  — OpenCV fast path + Pillow fallback for HEIC/HEIF/AVIF
-  — appends to buffers and fires _broadcast_update()

---

### pipeline.py — SmartCityPipeline

Owns every model. process_frame() is the single call that runs the full AI pipeline on one BGR frame.

Constructor args:
- camera_id=cam_01, vehicle_model_size=yolov8m, face_ctx_id=-1 (CPU)
- face_sim_threshold=0.70, body_sim_threshold=0.82, body_match_min_margin=0.06
- anpr_interval=1 (every frame), inference_max_side=0 (disabled)
- face_det_size=320 (base, adaptive_detect scales up)
- enhance_gamma=True, enhance_clahe=True, enhance_zero_dce=True
- gamma_target_mean=128.0, clahe_clip_limit=3.0
- plate_model_path=weights/license_plate_yolov8n.pt
- n_threads=4

Models loaded at startup:
1. PersonReIdentifier (OSNet x1_0)
2. PersonRegistry (pgvector, PostgreSQL)
3. DayNightClassifier (MobileNetV2)
4. VehicleDetector (YOLOv8m + BotSORT)
5. FaceRecogniser (InsightFace buffalo_l)
6. ZeroDCEEnhancer (Gamma+CLAHE+Zero-DCE++)
7. PersonDetector (YOLOv8m + BotSORT)
8. ANPRService (YOLOv11n + EasyOCR)
9. DehazingService (DCP+MSRCR)
10. ThreadPoolExecutor (4 workers for parallel vehicle detection)

FrameEvent dataclass fields:
- frame_id, camera_id, timestamp, day_night (dict), enhanced (bool)
- vehicles (VehicleDetectionResult), persons (PersonDetectionResult)
- plates (ANPRResult), faces (list of dicts), identified_people (list)
- alerts (list of face_watchlist_hit + person_registry_hit dicts)

---

### pg_vector.py — PersonRegistry

Single source of truth for all person identity. Backed by PostgreSQL + pgvector extension.

Database tables (created by pgvector_setup.sql):
- persons(person_id UUID, name TEXT)
- face_embeddings(id, person_id, embedding vector(512))
- body_embeddings(id, person_id, embedding vector(2048))

Key methods:
- add_person(name, images, face_recogniser, person_detector, reidentifier) → RegistrationOutcome
  - Deduplicates by name (case-insensitive) before creating new person_id
  - Extracts ArcFace embedding (face) + OSNet embedding (body) per image
- match_face(embedding) → (person_dict, similarity) — cosine via pgvector <=> operator
- match_body(embedding, crop_shape) — TOP-2 neighbours + margin check (rejects ambiguous matches)
- identify(face_embedding, body_embedding, body_crop_shape) — face-first priority
  - If face found but no match → returns None (does NOT fall back to body)
  - Body matching only fires when face_embedding is None entirely
- list_people() → list with face_refs and body_refs counts
- remove(person_id) → bool

Thresholds:
- face_sim_threshold=0.70 (cosine similarity)
- body_sim_threshold=0.82 (raised from 0.50 to reduce false positives)
- body_match_min_margin=0.06 (gap between best and second-best body match)

---

### clashifiers/identity_resolver.py — IdentityResolver

Stateless per-frame identity resolver used by POST /analyse/identify. Centralises the face+body binding logic so the REST endpoint and the live pipeline use identical logic.

Steps per frame:
1. PersonDetector.detect() → body bboxes
2. FaceRecogniser.adaptive_detect() → face detections
3. Spatial face→body binding (face centre inside body bbox, greedy by confidence)
4. Body Re-ID embeddings (only if enable_body_matching=True, default=False)
5. PersonRegistry.identify() per bound person (face-first)
6. Unbound faces (face detected but no body bbox matched) matched separately

Output: { frame_id, camera_id, person_count, people[], unbound_faces[] }

---

### clashifiers/day_night/main.py — DayNightClassifier

Two-stage classifier:
1. Brightness heuristic (instant, no model):
   - mean >= 160 → day, skip enhancement
   - mean <= 85  → night, route to enhancement  (DARK_THRESH raised from 60→85)
   - 86–159      → MobileNetV2 CNN decides
2. MobileNetV2 CNN (1280→256→2 head, ImageNet pretrained, not fine-tuned)

Output: { label, confidence, route_to_enhancement, method }

---

### clashifiers/face_recognization/main.py — FaceRecogniser

Three-stage pipeline:
1. _preprocess_for_face_detection() — CLAHE on LAB L-channel (always-on, ~0.5ms)
2. RetinaFace — face detection + 5 landmarks (det_thresh=0.35, was 0.5)
3. ArcFace — 512-d identity embedding per detected face

adaptive_detect(frame, person_count_hint):
- 1-2 people → 320px detection grid
- 3-8 people → 640px grid
- 9+ people  → 960px grid
- Retries upward if no faces found at starting size
- _set_det_size() caches current size → only calls app.prepare() when size actually changes (fixes log spam)

Watchlist class (in-memory, legacy) — brute-force cosine search. Production: use pgvector PersonRegistry.

---

### clashifiers/person_detector/main.py — PersonDetector

YOLOv8m filtering COCO class 0 (person) with BotSORT tracking.

Enhancements:
- conf=0.30 (was 0.4), iou=0.50 (was 0.45)
- tracker=botsort.yaml (was ByteTrack — better occlusion handling)
- MIN_PERSON_HEIGHT=40px filter in _parse() (kills shadow/bag false positives)
- adaptive_detect(frame, frame_id, camera_id, person_count_hint):
  - 0-3 → native resolution (fast path)
  - 4-8 → upscale to 1280px wide before detection, scale bboxes back
  - 9+  → upscale to 1920px wide

Dataclasses:
- PersonDetection: bbox, confidence, track_id, frame_id, center, area, width, height
- PersonDetectionResult: frame_id, camera_id, person_count, detections[]

---

### clashifiers/person_reid/main.py — PersonReIdentifier

OSNet x1_0 (torchreid) for body appearance embedding.
- embed(crop) → 2048-dim float32 numpy vector
- Only called when no face found in person crop (face-first architecture)
- Used during registration (add_person) and live pipeline (process_frame)

---

### clashifiers/vechile_detector/main.py — VehicleDetector

YOLOv8m with BotSORT tracking, filtering COCO vehicle classes:
- 2=car, 3=motorcycle, 5=bus, 7=truck

Dataclasses:
- VehicleDetection: bbox, label, confidence, class_id, track_id, center, area
- VehicleDetectionResult: frame_id, camera_id, total, vehicle_count (per-class), detections[]

---

### services/classical_enhance.py — Classical Enhancement

CPU-only, model-free building blocks:
- gamma_correction(frame, gamma) — 256-entry LUT
- estimate_gamma(frame, target_mean=128.0) — adaptive gamma from current brightness
- clahe_enhance(frame, clip_limit=3.0, tile_grid_size=(8,8)) — LAB L-channel CLAHE
- auto_enhance(frame) — gamma + CLAHE in one call

---

### services/zero_dce.py — ZeroDCEEnhancer

Chains three enhancement stages in order, each feeding the next:
1. Gamma Correction (adaptive, target_mean=128.0)
2. CLAHE (clip=3.0, tile=8x8, LAB L-channel)
3. Zero-DCE++ CNN (Epoch99.pth, scale_factor=12)

Any stage can be disabled via constructor flags. Zero-DCE++ stage auto-skips if weights missing.

Properties: method, last_stages_applied, last_gamma

---

### services/anpr.py — ANPRService

Two-stage plate recognition:
Stage 1 — Localisation:
- YOLOv11n fine-tuned plate detector (if weights present)
- Contour heuristic fallback (Canny → contour → aspect-ratio filter 1.5–6.0)
Stage 2 — OCR:
- EasyOCR (preferred, ~100MB download on first use)
- pytesseract fallback
- ROI preprocessing: 2× upscale + sharpening + Otsu threshold
- cleaned_text = uppercase alphanumeric only (e.g. MH12AB1234)

Restricted to vehicle bounding boxes in pipeline to reduce false positives.

Dataclasses:
- PlateReading: bbox, raw_text, cleaned_text, confidence, frame_id
- ANPRResult: frame_id, camera_id, plate_count, ocr_engine, plates[]

---

### services/dehazing.py — DehazingService

Auto-selects algorithm based on frame brightness:
- mean <= 0.72 → Sky-Aware DCP (Dark Channel Prior)
  - 15×15 dark channel → sky detection → adaptive omega (0.90 ground, 0.50 sky)
  - Guided filter → white balance → gamma(0.85) → contrast stretch → unsharp mask
- mean > 0.72 → MSRCR (Multi-Scale Retinex with Colour Restoration)
  - Retinex at sigmas [15, 80, 250] → colour restoration → percentile normalise

Dataclass: DehazingResult: dehazed_frame, transmission, atm_light, method

---

### Zero-DCE_extension/Zero-DCE++/ — External Repo

Cloned external repo. Do not modify directly.

| File | Purpose |
|---|---|
| model.py | enhance_net_nopool — 7-layer depthwise-sep CNN with U-Net skip connections |
| Myloss.py | L_color, L_spa, L_exp, L_TV unsupervised loss functions |
| dataloader.py | lowlight_loader — 512×512 low-light image dataset |
| lowlight_train.py | Training (CUDA, 100 epochs) |
| lowlight_test.py | Inference on test_data/ folder |
| snapshots_Zero_DCE++/Epoch99.pth | Pretrained weights used by ZeroDCEEnhancer |

---

## 6. Import Graph

```
main.py
  ├── pipeline.py
  │     ├── clashifiers/day_night/main.py           → DayNightClassifier
  │     ├── clashifiers/vechile_detector/main.py    → VehicleDetector
  │     ├── clashifiers/face_recognization/main.py  → FaceRecogniser
  │     ├── clashifiers/person_detector/main.py     → PersonDetector
  │     ├── clashifiers/person_reid/main.py         → PersonReIdentifier
  │     ├── pg_vector.py                            → PersonRegistry
  │     ├── services/zero_dce.py                    → ZeroDCEEnhancer
  │     ├── services/anpr.py                        → ANPRService
  │     └── services/dehazing.py                    → DehazingService
  └── clashifiers/identity_resolver.py              → IdentityResolver

services/zero_dce.py
  ├── services/classical_enhance.py
  └── Zero-DCE_extension/Zero-DCE++/model.py
```

---

## 7. Alert Types (WebSocket + /alerts)

| Alert type | When fired | Key fields |
|---|---|---|
| face_watchlist_hit | Face ArcFace embedding matches a registered person above face_sim_threshold | type, person_id, name, similarity, frame_id, camera_id, timestamp |
| person_registry_hit | Body or face binding in process_frame matches via PersonRegistry.identify() | type, person_id, name, similarity, method (face/body), track_id, frame_id |

---

## 8. Running the Server

```bash
cd py-server
source .venv/bin/activate
pip install -r requirements.txt

# PostgreSQL + pgvector must be running
psql -U postgres -d smart_city -f pgvector_setup.sql

# Start server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Swagger UI
# http://localhost:8000/docs
```

---

## 9. Key Design Decisions

| Decision | Rationale |
|---|---|
| Single source of truth: pgvector PersonRegistry | Old Watchlist (in-memory) and PersonRegistry disagreed on same photo. Now one store for face+body identity. |
| Face-first, body-fallback identity | If face detected but no match → Unknown. Never fall back to body matching when face visible — body Re-ID has high FP rate |
| Body margin check (top-2 neighbours) | Single threshold cannot separate true matches from false positives in group photos. Margin check rejects ambiguous results |
| Always-on CLAHE before face detection | Day/Night classifier misses slightly dark frames (indoor, shade). CLAHE runs ~0.5ms on CPU, improves RetinaFace recall without waiting for night classification |
| DARK_THRESH raised 60→85 | Frames 61–85 mean brightness (indoor lighting) now route to full enhancement chain instead of the untrained CNN |
| BotSORT over ByteTrack | BotSORT maintains track IDs through occlusion (person behind pole/car) more reliably |
| adaptive_detect for persons + faces | Upscales frame before detection when crowd is dense — catches small/distant people that YOLO misses at native resolution |
| Min-height filter 40px | Prevents bags, shadows, partial blobs from being counted as persons at the lower conf threshold |
| ANPR restricted to vehicle boxes | Prevents false plate reads from signs, graffiti, building numbers in background |
| ThreadPoolExecutor for vehicle detection | Vehicle detection runs in parallel with person+face detection — hides its latency |
| MJPEG + WebSocket dual output | MJPEG for video frames (img tag compatible), WebSocket for structured JSON events — both fed by same _stream_worker loop |

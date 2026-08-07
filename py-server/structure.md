# Smart City Platform — py-server Documentation

> **Last updated:** 2026-08-04 | **Server:** FastAPI + Uvicorn | **Python:** 3.12

---

## 1. What This Server Does

The py-server is the AI/ML backend of the Smart City Platform. It processes images and video streams from cameras, running multiple computer-vision pipelines per frame and returning structured JSON or annotated images over REST and WebSocket.

### Core Capabilities

| Capability | Models Used |
|---|---|
| Day/Night classification | MobileNetV2 (ImageNet pretrained, two-stage heuristic+CNN) |
| Night image enhancement | **SCI** (Self-Calibrated Illumination, CVPR 2022) + optional post-CLAHE — *default* |
| Night image enhancement (legacy) | Gamma Correction → CLAHE → Zero-DCE++ (selectable via `night_enhancement_backend="zero_dce"`) |
| Vehicle detection + tracking | YOLOv8m + BotSORT |
| Person detection + tracking | YOLOv8m + BotSORT (adaptive resolution) |
| Face detection + recognition | RetinaFace + ArcFace (InsightFace buffalo_l) |
| Person identity matching | pgvector (face embeddings + body Re-ID, face-first architecture) |
| Track-level face embedding pooling | Rolling mean of last N ArcFace embeddings per BotSORT track_id |
| Number plate reading (ANPR) | YOLOv11n (plate detector) + EasyOCR |
| Haze/fog removal | Dark Channel Prior (Sky-Aware) + MSRCR (auto-selected by brightness) |
| Live stream processing | RTSP/file via OpenCV background worker |
| Real-time WebSocket push | ws://localhost:8000/ws |

---

## 2. Directory Tree

```
py-server/
├── main.py                              ← FastAPI app — all REST + WebSocket endpoints
├── pipeline.py                          ← SmartCityPipeline — orchestrates all models per frame
├── pg_vector.py                         ← PersonRegistry — pgvector-backed face+body identity store
├── pgvector_setup.sql                   ← SQL to create persons/person_images/face_embeddings/body_embeddings tables
├── requirements.txt                     ← pip dependencies
├── download_weights.py                  ← Helper to download model weight files
├── yolov8m.pt                           ← YOLOv8m weights (COCO, shared by vehicle+person detectors)
├── yolov8n.pt                           ← YOLOv8n weights (lighter variant)
├── structure.md                         ← This documentation file
├── run.txt                              ← Quick-start run commands
├── .env                                 ← Environment variables (DB DSN etc.)
├── .venv/                               ← Python virtual environment
├── weights/
│   ├── license_plate_yolov8n.pt         ← Fine-tuned plate detector (YOLOv11n)
│   ├── sci_difficult/                   ← SCI weights trained on DARK FACE dataset (default)
│   ├── sci_medium/                      ← SCI weights trained on LOL + LSRW (indoor/dusk)
│   └── sci_easy/                        ← SCI weights trained on MIT-Adobe FiveK (mild low-light)
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
│   ├── sci_enhance.py                   ← SCIEnhancer — Self-Calibrated Illumination (CVPR 2022), DEFAULT backend
│   ├── classical_enhance.py             ← Gamma Correction + CLAHE (CPU-only, model-free)
│   ├── zero_dce.py                      ← ZeroDCEEnhancer — chains Gamma→CLAHE→Zero-DCE++ (legacy backend)
│   ├── anpr.py                          ← ANPRService — plate localisation + EasyOCR
│   └── dehazing.py                      ← DehazingService — Sky-Aware DCP + MSRCR
│
└── Zero-DCE_extension/
    └── Zero-DCE++/
        ├── model.py                     ← enhance_net_nopool CNN architecture
        ├── Myloss.py                    ← L_color, L_spa, L_exp, L_TV loss functions
        ├── dataloader.py                ← lowlight_loader PyTorch dataset
        ├── lowlight_train.py            ← Training script (CUDA, 100 epochs)
        ├── lowlight_test.py             ← Inference script
        └── snapshots_Zero_DCE++/
            └── Epoch99.pth              ← Pretrained weights used by ZeroDCEEnhancer (legacy)
```

---

## 3. API Routes

**Base URL:** http://localhost:8000  |  **Swagger:** http://localhost:8000/docs  |  **WebSocket:** ws://localhost:8000/ws

### Health

| Method | Route | Description |
|---|---|---|
| GET | /status | Pipeline ready state, uptime, model names, active night enhancement backend |

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
| POST | /enhance/frame | Active enhancer (SCI or Zero-DCE++ chain) → enhanced JPEG + X-Enhancement-Method / X-Stages-Applied / X-Gamma-Used headers |
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
| POST | /dehaze/frame | Dehazed JPEG (strength param 0.5–1.0, default 0.90) |
| POST | /dehaze/frame/compare | Side-by-side comparison JPEG (hazy vs. dehazed) |

### Stream (RTSP / File)

| Method | Route | Description |
|---|---|---|
| POST | /stream/start | Start background stream worker (body: source, camera_id) |
| POST | /stream/stop | Stop the running stream |
| GET | /stream/status | running, source, camera_id, started_at, frames_processed, error |
| GET | /stream/mjpeg | MJPEG HTTP stream — consumed by img tag (~30fps). Streams a placeholder frame when no stream is active. |
| GET | /stream/frame | Latest single annotated JPEG (for polling). 503 if no active stream. |

### Watchlist / Person Registry

| Method | Route | Description |
|---|---|---|
| POST | /watchlist/add | Upload reference photo(s) → extract face+body embeddings → store in pgvector. Supports multi-photo upload, name deduplication, optional `person_id`, `image_url`, `image_urls` (JSON list), and `night_augment` flag. |
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
  "stream_status": { "running": true, "source": "...", "camera_id": "cam_01", "started_at": 1234567890.0, "frames_processed": 42, "error": null },
  "latest_event":  { "...full FrameEvent dict..." },
  "alerts":        [ "...all accumulated alerts..." ]
}
```

---

## 4. Pipeline Flow (process_frame)

Every frame goes through these stages in order inside `SmartCityPipeline.process_frame()`:

```
BGR Frame
  │
  ▼
[0] _prune_track_history()
    Drop BotSORT track embedding histories not seen in last 150 frames

  ▼
[1] DayNightClassifier.predict()
    BRIGHT_THRESH=160 → day (skip enhancement)
    DARK_THRESH=85    → night → enhancement
    86-159 brightness → MobileNetV2 CNN decides
  │
  ├── if night ──▶ [2] SCIEnhancer.enhance()           ← DEFAULT backend
  │                     SCI EnhanceNetwork (illumination-estimation block)
  │                     → optional post-CLAHE (clip=2.0, tile=8x8, LAB L-channel)
  │               OR ZeroDCEEnhancer.enhance()         ← legacy, opt-in
  │                     Gamma Correction (adaptive, target mean=128)
  │                     → CLAHE (clip=3.0, tile=8x8, LAB L-channel)
  │                     → Zero-DCE++ (CNN, Epoch99.pth)
  │
  ▼ (working frame = enhanced or original)
  │
  ├── [_maybe_downscale()] if inference_max_side > 0   ← disabled by default (=0)
  │
  ├──[parallel]──▶ [3] VehicleDetector.detect()        ← submitted to ThreadPoolExecutor
  │
  ├──[sync]──────▶ [4] PersonDetector.adaptive_detect()
  │                     0-3 people  → native resolution
  │                     4-8 people  → upscale to 1280px wide
  │                     9+ people   → upscale to 1920px wide
  │                     min_height filter=40px
  │                     BotSORT tracker, conf=0.30, iou=0.50
  │
  ├──[sync]──────▶ [5] FaceRecogniser.adaptive_detect()
  │                     _preprocess_for_face_detection() ← CLAHE always-on
  │                     det_thresh=0.35
  │                     320px grid for 1-2 people
  │                     640px grid for 3-8 people
  │                     960px grid for 9+ people (retry upward if no faces found)
  │                     ArcFace 512-d embedding per face
  │
  ├──[wait]──────▶ [6] VehicleDetector result collected (fut_vehicles.result())
  │
  ├──────────────▶ [7] Spatial face→body binding + track-level embedding aggregation
  │                     For each person detection: find face whose centre is inside body bbox
  │                     If face found → _update_track_face_history(track_id, embedding)
  │                       Rolling pool of last face_embedding_history_len=5 embeddings
  │                       Guard: resets history if new face cosine sim < 0.40 vs pooled
  │                       (catches BotSORT track-ID handoffs to a different person)
  │                     face_query_embedding = aggregated (pooled) embedding
  │
  ├──────────────▶ [8] Face identity matching via PersonRegistry.match_face()
  │                     pgvector cosine similarity (<=>) operator
  │                     Uses aggregated embedding if track-bound, else raw single-frame embedding
  │                     face_sim_threshold=0.20
  │                     → face_watchlist_hit alert if matched
  │
  ├──────────────▶ [9] Per-person body+face binding + identity
  │                     For each detected person body:
  │                       - If face found in body bbox → use face embedding for identity
  │                         (body Re-ID embed() call is SKIPPED to save cost)
  │                       - If no face → run OSNet body Re-ID embedding
  │                     PersonRegistry.identify() → face-first, body fallback
  │                     body_sim_threshold=0.22, margin_check=0.06
  │                     → person_registry_hit alert if matched
  │
  └──────────────▶ [10] ANPRService.read_plates_in_vehicles()
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

Entry point. Loads `SmartCityPipeline` once at startup via `lifespan()`. Routes all HTTP and WS requests.

**Key globals:**
- `pipeline` — singleton `SmartCityPipeline` instance
- `event_buffer` — `deque(maxlen=500)` of FrameEvent dicts
- `alert_buffer` — `deque(maxlen=200)` of alert dicts
- `_ws_clients` — `set[WebSocket]` of active WebSocket connections
- `stream_state` — dict tracking running/source/camera_id/started_at/frame_count/error
- `_latest_annotated_frame` — bytes of last annotated JPEG (written by stream worker, read by /stream/mjpeg)

**Key functions:**
- `_stream_worker(source, camera_id)` — async background loop: reads frames from `cv2.VideoCapture`, calls `process_frame`, annotates, caches JPEG. Video files loop on EOF.
- `_mjpeg_generator()` — async generator yielding multipart MJPEG chunks at ~30fps. Streams a black placeholder frame when idle.
- `_broadcast_update()` — pushes `_ws_snapshot()` to all connected WS clients
- `_ws_snapshot()` — builds `{stream_status, latest_event, alerts}` dict
- `_decode_bytes(raw)` — OpenCV fast path + Pillow fallback for HEIC/HEIF/AVIF/GIF
- `_store(event)` — appends to buffers and fires `_broadcast_update()`
- `_extract_vehicle_boxes(vehicle_result)` — defensive helper that extracts bbox list from `VehicleDetectionResult` regardless of attribute name

**Pipeline startup args (lifespan):**
- `night_enhancement_backend="sci"` — selects SCIEnhancer as default (set to `"zero_dce"` to revert)
- `anpr_interval=1` — ANPR runs on every frame
- `inference_max_side=0` — downscaling disabled to preserve plate details
- `plate_model_path="weights/license_plate_yolov8n.pt"`
- `anpr_min_confidence=0.10`

---

### pipeline.py — SmartCityPipeline

Owns every model. `process_frame()` is the single call that runs the full AI pipeline on one BGR frame.

**Constructor args:**
- `camera_id="cam_01"`, `vehicle_model_size="yolov8m"`, `face_ctx_id=-1` (CPU)
- `face_sim_threshold=0.20`, `body_sim_threshold=0.22`, `body_match_min_margin=0.06`
- `anpr_interval=1` (every frame), `inference_max_side=0` (disabled)
- `face_det_size=320` (base, `adaptive_detect` scales up for crowds)
- `night_enhancement_backend="sci"` — `"sci"` (default) or `"zero_dce"`
- `sci_weights_path=None` (defaults to `weights/sci_difficult.pt`), `sci_device="cpu"`, `sci_n_threads=4`, `sci_enable_post_clahe=True`
- `enhance_gamma=True`, `enhance_clahe=True`, `enhance_zero_dce=True` (only used when backend is `"zero_dce"`)
- `gamma_target_mean=128.0`, `clahe_clip_limit=3.0`
- `plate_model_path=None`, `anpr_min_confidence=0.10`
- `n_threads=4` (ThreadPoolExecutor workers)
- `face_embedding_history_len=5` — frames of face embeddings to pool per track
- `track_history_prune_after=150` — evict a track's embedding history after this many frames unseen

**Models loaded at startup (in order):**
1. `PersonReIdentifier` (OSNet x1_0, CPU)
2. `PersonRegistry` (pgvector, PostgreSQL DSN)
3. `DayNightClassifier` (MobileNetV2)
4. `VehicleDetector` (YOLOv8m + BotSORT)
5. `FaceRecogniser` (InsightFace buffalo_l, RetinaFace+ArcFace)
6. `SCIEnhancer` **or** `ZeroDCEEnhancer` (selected by `night_enhancement_backend`)
7. `PersonDetector` (YOLOv8m + BotSORT)
8. `ANPRService` (YOLOv11n + EasyOCR)
9. `DehazingService` (Sky-Aware DCP + MSRCR)
10. `ThreadPoolExecutor` (4 workers for parallel vehicle detection)

**FrameEvent dataclass fields:**
- `frame_id`, `camera_id`, `timestamp`, `day_night` (dict), `enhanced` (bool)
- `vehicles` (VehicleDetectionResult), `persons` (PersonDetectionResult)
- `plates` (ANPRResult), `faces` (list of dicts), `identified_people` (list)
- `alerts` (list of face_watchlist_hit + person_registry_hit dicts)

**Track-level face embedding aggregation (`_update_track_face_history`):**
- Maintains a per-track rolling buffer of the last `face_embedding_history_len` (default 5) L2-normalized ArcFace embeddings.
- Resets history if new face cosine similarity to existing pooled embedding < 0.40 (BotSORT track-ID handoff guard).
- Returns the mean of the normalized buffer, renormalized — used for both `match_face()` and `PersonRegistry.identify()` so `/events faces` and `identified_people` always agree.

---

### pg_vector.py — PersonRegistry

Single source of truth for all person identity. Backed by PostgreSQL + pgvector extension.

**Database tables** (created by pgvector_setup.sql):
- `persons(person_id TEXT PK, name TEXT, image_url TEXT)` — primary display photo URL
- `person_images(id SERIAL PK, person_id TEXT FK, image_url TEXT, position INT)` — all reference photo URLs; `position=0` mirrors `persons.image_url`
- `face_embeddings(id, person_id, embedding vector(512))` — ArcFace 512-d embeddings (daylight + night-domain variants)
- `body_embeddings(id, person_id, embedding vector(512))` — OSNet Re-ID embeddings

**Key methods:**
- `add_person(name, images, face_recogniser, person_detector, reidentifier, person_id=None, enhancer=None, night_augment=True, image_url=None, image_urls=None)` → `RegistrationOutcome`
  - Deduplicates by name (case-insensitive) before creating a new `person_id`
  - Extracts ArcFace embedding (face) + OSNet embedding (body) per image
  - **Night-domain augmentation** (if `enhancer` is passed and `night_augment=True`): each reference image is synthetically darkened (`factor=0.35`, desaturated to grayscale + Gaussian noise `std=4.0`) and run through the same enhancement chain used at inference time, then embedded and stored as an additional `face_embeddings` row — fixes "known person shows Unknown at night"
  - Stores `image_urls` into `person_images` table (position-ordered)
- `match_face(embedding)` → `(person_dict, similarity)` — cosine via pgvector `<=>` operator
- `match_body(embedding, crop_shape)` — TOP-2 neighbours + margin check (rejects ambiguous matches)
- `identify(face_embedding, body_embedding, body_crop_shape)` — face-first priority
  - If `face_embedding` is provided (even if no match) → body matching is SKIPPED entirely
  - Body matching only fires when `face_embedding is None` (face not detected in crop)
- `list_people()` → list with `face_refs`, `body_refs`, and `image_urls` counts
- `remove(person_id)` → bool

**Thresholds:**
- `face_sim_threshold=0.20` (cosine similarity)
- `body_sim_threshold=0.22`
- `body_match_min_margin=0.06` (gap between best and second-best body match)

**`RegistrationOutcome` dataclass fields:**
- `person_id`, `name`, `images_received`, `face_embeddings_added`, `body_embeddings_added`
- `images_skipped`, `errors`, `registry_unavailable`, `reused_existing_person`
- `night_variants_added` — count of synthetic night-domain embeddings added per registration call

---

### clashifiers/identity_resolver.py — IdentityResolver

Stateless per-frame identity resolver used by `POST /analyse/identify`. Centralises the face+body binding logic so the REST endpoint and the live pipeline use identical logic.

Steps per frame:
1. `PersonDetector.detect()` → body bboxes
2. `FaceRecogniser.adaptive_detect()` → face detections
3. Spatial face→body binding (face centre inside body bbox, greedy by confidence)
4. Body Re-ID embeddings (only if `enable_body_matching=True`, default=`False`)
5. `PersonRegistry.identify()` per bound person (face-first)
6. Unbound faces (face detected but no body bbox matched) matched separately

Output: `{ frame_id, camera_id, person_count, people[], unbound_faces[] }`

---

### clashifiers/day_night/main.py — DayNightClassifier

Two-stage classifier:
1. Brightness heuristic (instant, no model):
   - mean >= 160 → day, skip enhancement
   - mean <= 85  → night, route to enhancement  (DARK_THRESH raised from 60→85)
   - 86–159      → MobileNetV2 CNN decides
2. MobileNetV2 CNN (1280→256→2 head, ImageNet pretrained, not fine-tuned)

Output: `{ label, confidence, route_to_enhancement, method }`

---

### clashifiers/face_recognization/main.py — FaceRecogniser

Three-stage pipeline:
1. `_preprocess_for_face_detection()` — CLAHE on LAB L-channel (always-on, ~0.5ms)
2. RetinaFace — face detection + 5 landmarks (`det_thresh=0.35`, was 0.5)
3. ArcFace — 512-d identity embedding per detected face

`adaptive_detect(frame, person_count_hint)`:
- 1-2 people → 320px detection grid
- 3-8 people → 640px grid
- 9+ people  → 960px grid
- Retries upward if no faces found at starting size
- `_set_det_size()` caches current size → only calls `app.prepare()` when size actually changes

`Watchlist` class (in-memory, legacy) — brute-force cosine search. Production: use pgvector `PersonRegistry`.

---

### clashifiers/person_detector/main.py — PersonDetector

YOLOv8m filtering COCO class 0 (person) with BotSORT tracking.

Enhancements:
- `conf=0.30` (was 0.4), `iou=0.50` (was 0.45)
- `tracker=botsort.yaml` (was ByteTrack — better occlusion handling)
- `MIN_PERSON_HEIGHT=40px` filter in `_parse()` (kills shadow/bag false positives)
- `adaptive_detect(frame, frame_id, camera_id, person_count_hint)`:
  - 0-3 → native resolution (fast path)
  - 4-8 → upscale to 1280px wide before detection, scale bboxes back
  - 9+  → upscale to 1920px wide

Dataclasses:
- `PersonDetection`: `bbox, confidence, track_id, frame_id, center, area, width, height`
- `PersonDetectionResult`: `frame_id, camera_id, person_count, detections[]`

---

### clashifiers/person_reid/main.py — PersonReIdentifier

OSNet x1_0 (torchreid) for body appearance embedding.
- `embed(crop)` → 512-d float32 numpy vector
- Only called when no face found in person crop (face-first architecture — body Re-ID embed() call is SKIPPED entirely when a face is found, saving inference cost)
- Used during registration (`add_person`) and live pipeline (`process_frame`)

---

### clashifiers/vechile_detector/main.py — VehicleDetector

YOLOv8m with BotSORT tracking, filtering COCO vehicle classes:
- 2=car, 3=motorcycle, 5=bus, 7=truck

Dataclasses:
- `VehicleDetection`: `bbox, label, confidence, class_id, track_id, center, area`
- `VehicleDetectionResult`: `frame_id, camera_id, total, vehicle_count` (per-class), `detections[]`

---

### services/sci_enhance.py — SCIEnhancer *(DEFAULT night backend)*

Self-Calibrated Illumination enhancer (Ma et al., CVPR 2022).

**Why SCI over Zero-DCE++:**
- Zero-DCE++ optimises purely for perceptual quality (human viewing), not downstream detector performance.
- SCI was explicitly benchmarked on low-light **face detection** and nighttime segmentation — the exact failure mode in this pipeline (person detection recall dropping at night, preventing face→body binding).
- At inference, only the `EnhanceNetwork` (illumination-estimation block) runs; the `CalibrateNetwork` is discarded after training. Lighter and faster than Zero-DCE++'s deeper U-Net.

**CPU benchmarks (4 threads):**

| Resolution | Latency | Approx FPS |
|---|---|---|
| 480×270 | ~14 ms | ~73 fps |
| 960×540 | ~38 ms | ~26 fps |
| 1280×720 | ~181 ms | ~5.5 fps |

**Architecture:**
- `EnhanceNetwork(layers=1, channels=3)`: `in_conv` → 1 weight-shared residual conv block → `out_conv (Sigmoid)`
- Forward: `illumination = net(x)` → `enhanced = clamp(x / illumination, 0, 1)`
- Weights: `weights/sci_difficult.pt` (DARK FACE-trained, default), `sci_medium.pt`, `sci_easy.pt`

**Constructor params:**
- `weights_path` — path to `.pt` checkpoint (defaults to `weights/sci_difficult.pt`)
- `device` — `'cpu'` or `'cuda'`
- `n_threads` — torch CPU thread count
- `enable_post_clahe=True` — light CLAHE pass after SCI for extra local contrast
- `clahe_clip_limit=2.0`, `clahe_tile_grid_size=(8,8)`

**API parity with ZeroDCEEnhancer:**
- Same `.enhance(frame_bgr)` contract (BGR uint8 in/out)
- Same `.method`, `.last_stages_applied`, `.last_gamma` attributes
- Falls back to CLAHE-only if torch or weights are unavailable

---

### services/classical_enhance.py — Classical Enhancement

CPU-only, model-free building blocks:
- `gamma_correction(frame, gamma)` — 256-entry LUT
- `estimate_gamma(frame, target_mean=128.0)` — adaptive gamma from current brightness
- `clahe_enhance(frame, clip_limit=3.0, tile_grid_size=(8,8))` — LAB L-channel CLAHE
- `auto_enhance(frame)` — gamma + CLAHE in one call

---

### services/zero_dce.py — ZeroDCEEnhancer *(legacy, opt-in)*

Chains three enhancement stages in order, each feeding the next:
1. Gamma Correction (adaptive, `target_mean=128.0`)
2. CLAHE (`clip=3.0`, `tile=8x8`, LAB L-channel)
3. Zero-DCE++ CNN (`Epoch99.pth`, `scale_factor=12`)

Any stage can be disabled via constructor flags. Zero-DCE++ stage auto-skips if weights missing.

Properties: `method`, `last_stages_applied`, `last_gamma`

---

### services/anpr.py — ANPRService

Two-stage plate recognition:

**Stage 1 — Localisation:**
- YOLOv11n fine-tuned plate detector (if weights present at `weights/license_plate_yolov8n.pt`)
- Contour heuristic fallback (Canny → contour → aspect-ratio filter 1.5–6.0)

**Stage 2 — OCR:**
- EasyOCR (preferred, ~100MB download on first use)
- pytesseract fallback
- ROI preprocessing: 2× upscale + sharpening + Otsu threshold
- `cleaned_text` = uppercase alphanumeric only (e.g. `MH12AB1234`)

Restricted to vehicle bounding boxes in pipeline to reduce false positives.

Dataclasses:
- `PlateReading`: `bbox, raw_text, cleaned_text, confidence, frame_id`
- `ANPRResult`: `frame_id, camera_id, plate_count, ocr_engine, plates[]`

---

### services/dehazing.py — DehazingService

Auto-selects algorithm based on frame brightness:
- mean <= 0.72 → **Sky-Aware DCP** (Dark Channel Prior)
  - 15×15 dark channel → sky detection → adaptive omega (0.90 ground, 0.50 sky)
  - Guided filter → white balance → gamma(0.85) → contrast stretch → unsharp mask
- mean > 0.72 → **MSRCR** (Multi-Scale Retinex with Colour Restoration)
  - Retinex at sigmas [15, 80, 250] → colour restoration → percentile normalise

Dataclass: `DehazingResult`: `dehazed_frame, transmission, atm_light, method`

---

### Zero-DCE_extension/Zero-DCE++/ — External Repo

Cloned external repo. Do not modify directly.

| File | Purpose |
|---|---|
| model.py | `enhance_net_nopool` — 7-layer depthwise-sep CNN with U-Net skip connections |
| Myloss.py | L_color, L_spa, L_exp, L_TV unsupervised loss functions |
| dataloader.py | `lowlight_loader` — 512×512 low-light image dataset |
| lowlight_train.py | Training (CUDA, 100 epochs) |
| lowlight_test.py | Inference on test_data/ folder |
| snapshots_Zero_DCE++/Epoch99.pth | Pretrained weights used by ZeroDCEEnhancer (legacy backend) |

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
  │     ├── services/sci_enhance.py                 → SCIEnhancer        (DEFAULT)
  │     ├── services/zero_dce.py                    → ZeroDCEEnhancer    (legacy opt-in)
  │     ├── services/anpr.py                        → ANPRService
  │     └── services/dehazing.py                    → DehazingService
  └── clashifiers/identity_resolver.py              → IdentityResolver

services/sci_enhance.py
  └── (pure PyTorch + OpenCV, no local deps)

services/zero_dce.py
  ├── services/classical_enhance.py
  └── Zero-DCE_extension/Zero-DCE++/model.py
```

---

## 7. Alert Types (WebSocket + /alerts)

| Alert type | When fired | Key fields |
|---|---|---|
| `face_watchlist_hit` | Face ArcFace embedding matches a registered person above `face_sim_threshold`. Uses track-aggregated (pooled) embedding when face is bound to a BotSORT track. | `type, person_id, name, image_url, similarity, frame_id, camera_id, timestamp` |
| `person_registry_hit` | Body or face binding in `process_frame` matches via `PersonRegistry.identify()` | `type, person_id, name, image_url, similarity, method (face/body), track_id, frame_id, camera_id, timestamp` |

---

## 8. Database Schema

```sql
-- pgvector_setup.sql

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE persons (
    person_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    image_url   TEXT            -- primary display photo URL
);

CREATE TABLE person_images (
    id          SERIAL PRIMARY KEY,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    image_url   TEXT NOT NULL,
    position    INT  NOT NULL DEFAULT 0  -- 0 = primary, >0 = additional
);
-- Unique index: (person_id, image_url)

CREATE TABLE face_embeddings (
    id          SERIAL PRIMARY KEY,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    embedding   vector(512) NOT NULL     -- ArcFace 512-d (daylight + night-domain variants per person)
);

CREATE TABLE body_embeddings (
    id          SERIAL PRIMARY KEY,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    embedding   vector(512) NOT NULL     -- OSNet Re-ID
);

-- IVFFlat indexes for fast cosine similarity search
CREATE INDEX idx_face_embeddings_vec ON face_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
CREATE INDEX idx_body_embeddings_vec ON body_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
```

---

## 9. Running the Server

```bash
cd py-server
source .venv/bin/activate
pip install -r requirements.txt

# PostgreSQL + pgvector must be running
# sudo apt-get install -y postgresql-17-pgvector
psql -U postgres -d smart_city -f pgvector_setup.sql

# Download SCI weights (sci_difficult.pt → weights/)
python download_weights.py

# Start server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Swagger UI
# http://localhost:8000/docs
```

---

## 10. Key Design Decisions

| Decision | Rationale |
|---|---|
| **SCI replaces Zero-DCE++ as default night enhancer** | Zero-DCE++ was trained for perceptual quality, not detector performance. SCI (CVPR 2022) was benchmarked specifically on low-light face detection as a downstream task, directly matching the observed failure mode: person/face detection recall dropping at night. SCI is also lighter at inference (single illumination-estimation block; calibrator discarded after training). |
| **Night-domain enrollment augmentation** | Root cause of "known person shows Unknown at night": daylight reference embeddings are out-of-domain vs. night-enhanced query embeddings. Fix: synthetically darken + desaturate each reference photo and run it through the same enhancement chain at enrollment time, storing both daylight and night-enhanced embeddings in the registry. |
| **Track-level face embedding pooling** | A single frame's ArcFace embedding under night enhancement is noisy. BotSORT provides stable track_ids, so the last 5 embeddings per track are pooled (mean of L2-normalised vectors, renormalized). Reduces frame-to-frame identity flicker. Track-ID handoff guard resets history if cosine sim to pooled embedding drops below 0.40. |
| **Single source of truth: pgvector PersonRegistry** | Old in-memory Watchlist and PersonRegistry disagreed on the same photo. Now one store for face+body identity (`faces` in FrameEvent and `identified_people` both use the same track-aggregated embedding). |
| **Face-first, body-fallback identity** | If face detected but no match → Unknown. Never fall back to body matching when a face is visible — body Re-ID has high FP rate in group scenes. Body embedding extraction is also SKIPPED (not just ignored) when a face is found, saving OSNet inference cost. |
| **Body margin check (top-2 neighbours)** | Single threshold cannot separate true from false positives in group photos. Margin check rejects ambiguous results (gap between best and 2nd-best < 0.06). |
| **Always-on CLAHE before face detection** | Improves RetinaFace recall on slightly dark/indoor frames without waiting for night classification. ~0.5ms on CPU. |
| **DARK_THRESH raised 60→85** | Frames with mean brightness 61–85 (indoor lighting) now route to full enhancement chain instead of the untrained MobileNetV2 CNN. |
| **BotSORT over ByteTrack** | Maintains track IDs through occlusion more reliably. |
| **adaptive_detect for persons + faces** | Upscales frame before detection when crowd is dense — catches small/distant people that YOLO misses at native resolution. |
| **Min-height filter 40px** | Prevents bags, shadows, partial blobs from being counted as persons at the lower conf threshold. |
| **ANPR restricted to vehicle boxes** | Prevents false plate reads from signs, graffiti, building numbers in background. |
| **ThreadPoolExecutor for vehicle detection** | Vehicle detection runs in parallel with person+face detection — hides its latency. |
| **MJPEG + WebSocket dual output** | MJPEG for video frames (`<img>` tag compatible), WebSocket for structured JSON events — both fed by the same `_stream_worker` loop. |
| **Video file looping in stream worker** | On EOF, `cap.set(CAP_PROP_POS_FRAMES, 0)` loops the file so demo footage plays continuously without restarting the worker. |

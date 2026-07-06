export interface DayNightResult {
  label: "day" | "night";
  confidence: number;
  route_to_enhancement: boolean;
  method: "heuristic" | "cnn";
}

export interface VehicleDetection {
  bbox: [number, number, number, number];
  label: "car" | "motorcycle" | "bus" | "truck" | "bicycle";
  confidence: number;
  class_id: number;
  track_id: number;
  center: [number, number];
  area: number;
}

export interface VehicleDetectionResult {
  frame_id: string;
  camera_id: string;
  total: number;
  vehicle_count: Record<string, number>;
  detections: VehicleDetection[];
}

export interface PersonDetection {
  bbox: [number, number, number, number];
  confidence: number;
  track_id: number;
  frame_id: string;
  center: [number, number];
  area: number;
  width: number;
  height: number;
}

export interface PersonDetectionResult {
  frame_id: string;
  camera_id: string;
  person_count: number;
  detections: PersonDetection[];
}

export interface PlateReading {
  bbox: [number, number, number, number];
  raw_text: string;
  cleaned_text: string;
  confidence: number;
  frame_id: string;
}

export interface ANPRResult {
  frame_id: string;
  camera_id: string;
  plate_count: number;
  ocr_engine: "easyocr" | "pytesseract";
  plates: PlateReading[];
}

export interface WatchlistMatch {
  person_id: string;
  name: string;
  similarity: number;
  is_match: boolean;
}

export interface FaceResult {
  face: {
    bbox: [number, number, number, number];
    confidence: number;
    landmarks: number[][];
  };
  match: WatchlistMatch | null;
  alert: boolean;
}

export interface AlertEvent {
  type: "face_watchlist_hit" | "person_registry_hit";
  person_id: string;
  name: string;
  similarity: number;
  frame_id: string;
  camera_id: string;
  timestamp: number;
  // person_registry_hit extras
  method?: "face" | "body";
  track_id?: number;
}

export interface FrameEvent {
  frame_id: string;
  camera_id: string;
  timestamp: number;
  day_night: DayNightResult;
  enhanced: boolean;
  vehicles: VehicleDetectionResult;
  persons: PersonDetectionResult;
  plates: ANPRResult;
  faces: FaceResult[];
  identified_people: IdentifiedPerson[];
  alerts: AlertEvent[];
}

export interface WatchlistPerson {
  person_id: string;
  name: string;
}

export interface StreamStatus {
  running: boolean;
  source: string | null;
  camera_id: string | null;
  frames_processed: number;
  error: string | null;
}

export interface ServerStatus {
  status: "ready" | "loading";
  uptime: number;
  models: string[];
}

// ── Person Registry ────────────────────────────────────────────────

export interface RegistryPerson {
  person_id: string;
  name: string;
  face_refs: number;
  body_refs: number;
}

export interface AddPersonOutcome {
  person_id: string;
  name: string;
  images_received: number;
  face_embeddings_added: number;
  body_embeddings_added: number;
  images_skipped: number;
  status: "ok" | "warning_no_usable_images";
}

export interface IdentifiedPerson {
  track_id: number;
  person_id: string;
  name: string;
  similarity: number;
  method: "face" | "body";
}

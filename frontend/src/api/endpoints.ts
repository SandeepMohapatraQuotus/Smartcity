import { api } from "./client";
import type {
  DayNightResult,
  VehicleDetectionResult,
  PersonDetectionResult,
  ANPRResult,
  FrameEvent,
  AlertEvent,
  WatchlistPerson,
  StreamStatus,
  ServerStatus,
  RegistryPerson,
  AddPersonOutcome,
  IdentifyResult,
} from "./types";


function fileForm(file: File, extra?: Record<string, string>) {
  const fd = new FormData();
  fd.append("file", file);
  if (extra) for (const [k, v] of Object.entries(extra)) fd.append(k, v);
  return fd;
}

// ── Health ───────────────────────────────────────────────────────
export const getServerStatus = () =>
  api.get<ServerStatus>("/status").then((r) => r.data);

// ── Full Pipeline ─────────────────────────────────────────────────
export const analyseFrame = (file: File) =>
  api.post<FrameEvent>("/analyse/frame", fileForm(file)).then((r) => r.data);

export const analyseFrameAnnotated = (file: File): Promise<Blob> =>
  api
    .post("/analyse/frame/annotated", fileForm(file), { responseType: "blob" })
    .then((r) => r.data);

// ── Identity (new unified route: adaptive detection + spatial binding) ──
export const analyseIdentify = (file: File) =>
  api
    .post("/analyse/identify", fileForm(file))
    .then((r) => r.data as IdentifyResult);

export const analyseIdentifyAnnotated = (file: File): Promise<Blob> =>
  // /analyse/frame/annotated still draws correct face boxes (pipeline.annotate)
  api
    .post("/analyse/frame/annotated", fileForm(file), { responseType: "blob" })
    .then((r) => r.data);

// ── Classifiers ───────────────────────────────────────────────────
export const classifyDayNight = (file: File) =>
  api.post<DayNightResult>("/classify/day-night", fileForm(file)).then((r) => r.data);

export const enhanceFrame = (file: File): Promise<Blob> =>
  api
    .post("/enhance/frame", fileForm(file), { responseType: "blob" })
    .then((r) => r.data);

export const detectVehicles = (file: File) =>
  api.post<VehicleDetectionResult>("/detect/vehicles", fileForm(file)).then((r) => r.data);

export const detectPersons = (file: File) =>
  api.post<PersonDetectionResult>("/detect/persons", fileForm(file)).then((r) => r.data);

// ── ANPR ──────────────────────────────────────────────────────────
export const readPlates = (file: File) =>
  api.post<ANPRResult>("/anpr/read", fileForm(file)).then((r) => r.data);

export const readPlatesAnnotated = (file: File): Promise<Blob> =>
  api
    .post("/anpr/read/annotated", fileForm(file), { responseType: "blob" })
    .then((r) => r.data);

// ── Dehazing ──────────────────────────────────────────────────────
export const dehazeFrame = (file: File, strength = 0.85): Promise<Blob> =>
  api
    .post("/dehaze/frame", fileForm(file, { strength: String(strength) }), {
      responseType: "blob",
    })
    .then((r) => r.data);

export const dehazeCompare = (file: File): Promise<Blob> =>
  api
    .post("/dehaze/frame/compare", fileForm(file), { responseType: "blob" })
    .then((r) => r.data);

// ── Stream ────────────────────────────────────────────────────────
export const startStream = (source: string, camera_id = "cam_01") =>
  api.post("/stream/start", { source, camera_id }).then((r) => r.data);

export const stopStream = () => api.post("/stream/stop").then((r) => r.data);

export const getStreamStatus = () =>
  api.get<StreamStatus>("/stream/status").then((r) => r.data);

// ── Watchlist / Person Registry ──────────────────────────────────
export const addToWatchlist = (person_id: string, name: string, photo: File) =>
  api
    .post("/watchlist/add", fileForm(photo, { person_id, name }))
    .then((r) => r.data);

export const removeFromWatchlist = (person_id: string) =>
  api.delete(`/watchlist/${person_id}`).then((r) => r.data);

export const getWatchlist = () =>
  api.get<{ total: number; people: WatchlistPerson[] }>("/watchlist").then((r) => r.data.people);

// ── Events & Alerts ───────────────────────────────────────────────
export const getEvents = (limit = 50) =>
  api.get<{ total: number; events: FrameEvent[] }>(`/events?limit=${limit}`).then((r) => r.data.events);

export const getAlerts = (limit = 100) =>
  api.get<{ total: number; alerts: AlertEvent[] }>(`/alerts?limit=${limit}`).then((r) => r.data.alerts);

export const clearBuffers = () => api.delete("/events").then((r) => r.data);

// ── Person Registry ───────────────────────────────────────────────
// All three routes map to /watchlist/* — the backend now serves the
// pgvector registry through those endpoints (Watchlist was removed).
export const addRegistryPerson = (name: string, images: File[]) => {
  const fd = new FormData();
  fd.append("name", name);
  // Field MUST be 'files' to match the FastAPI route: files: List[UploadFile]
  for (const img of images) fd.append("files", img);
  return api.post<AddPersonOutcome>("/watchlist/add", fd).then((r) => r.data);
};

export const removeRegistryPerson = (person_id: string) =>
  api.delete(`/watchlist/${person_id}`).then((r) => r.data);

export const listRegistryPeople = () =>
  api.get<{ total: number; people: RegistryPerson[] }>("/watchlist").then((r) => r.data.people);

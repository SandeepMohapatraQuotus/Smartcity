export const API_BASE_URL =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://localhost:8000";

export const MJPEG_STREAM_URL =
  (import.meta.env.VITE_MJPEG_STREAM_URL as string | undefined) ??
  `${API_BASE_URL}/stream/mjpeg`;

export const POLL_EVENTS_MS = Number(import.meta.env.VITE_POLL_EVENTS_MS ?? 800);
export const POLL_STATUS_MS = Number(import.meta.env.VITE_POLL_STATUS_MS ?? 5000);
export const POLL_STREAM_MS = 2000;
export const POLL_ALERTS_MS = 800;

export const DEFAULT_CAMERA_ID = "cam_01";

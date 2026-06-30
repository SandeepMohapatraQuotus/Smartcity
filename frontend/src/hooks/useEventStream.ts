/**
 * useEventStream — backed by WebSocket.
 * Keeps an in-memory rolling buffer of the last `limit` events received
 * over the WebSocket so existing consumers (events page, dashboard) don't
 * need to change their API.
 */
import { useEffect, useRef, useState } from "react";
import { useWebSocket } from "./useWebSocket";
import type { FrameEvent } from "@/api/types";

interface Options {
  enabled: boolean;
  limit?: number;
  /** @deprecated – no longer used; kept for API compat */
  interval?: number;
}

export function useEventStream({ enabled, limit = 50 }: Options) {
  const { latestEvent } = useWebSocket(enabled);
  const [events, setEvents] = useState<FrameEvent[]>([]);
  const lastIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (!latestEvent) return;
    // Deduplicate: only push when frame_id changes
    if (latestEvent.frame_id === lastIdRef.current) return;
    lastIdRef.current = latestEvent.frame_id;
    setEvents((prev) => {
      const next = [...prev, latestEvent];
      return next.length > limit ? next.slice(next.length - limit) : next;
    });
  }, [latestEvent, limit]);

  // Reset buffer when disabled
  useEffect(() => {
    if (!enabled) {
      setEvents([]);
      lastIdRef.current = null;
    }
  }, [enabled]);

  const latest = events.length ? events[events.length - 1] : null;
  return { events, latest };
}

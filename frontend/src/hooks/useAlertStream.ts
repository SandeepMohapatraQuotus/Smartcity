/**
 * useAlertStream — backed by WebSocket.
 * Tracks newly-seen alert keys exactly like the old polling version so
 * newCount increments are preserved.
 */
import { useEffect, useRef, useState } from "react";
import { useWebSocket } from "./useWebSocket";
import type { AlertEvent } from "@/api/types";

interface Options {
  enabled: boolean;
  limit?: number;
  /** @deprecated – no longer used; kept for API compat */
  interval?: number;
}

const keyOf = (a: AlertEvent) => `${a.person_id}::${a.frame_id}`;

export function useAlertStream({ enabled }: Options) {
  const { alerts: wsAlerts } = useWebSocket(enabled);
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  const [newCount, setNewCount] = useState(0);
  const seen = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!enabled || !wsAlerts.length) return;

    const deduped: AlertEvent[] = [];
    const keys = new Set<string>();
    for (const a of wsAlerts) {
      const k = keyOf(a);
      if (keys.has(k)) continue;
      keys.add(k);
      deduped.push(a);
    }

    let fresh = 0;
    for (const k of keys) if (!seen.current.has(k)) fresh++;
    seen.current = keys;

    setAlerts(deduped);
    if (fresh > 0) setNewCount((c) => c + fresh);
  }, [wsAlerts, enabled]);

  // Reset on disable
  useEffect(() => {
    if (!enabled) {
      setAlerts([]);
      setNewCount(0);
      seen.current = new Set();
    }
  }, [enabled]);

  return { alerts, newCount };
}

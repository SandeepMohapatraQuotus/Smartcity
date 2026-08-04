/**
 * useAlertStream — backed by WebSocket.
 *
 * The server pushes the FULL cumulative alert buffer on every WebSocket tick,
 * not just the delta. This means on tick N we receive all alerts from frames
 * 1..N. We must track which (person_id, frame_id) pairs we've already counted
 * so each unique detection is only counted ONCE regardless of how many ticks
 * arrive.
 *
 * Result: one DeduplicatedAlert card per person_id, with:
 *   - hits     = number of distinct frames the person was seen in
 *   - frame_id = most recent frame
 *   - first_frame_id = first frame this session
 */
import { useEffect, useRef, useState } from "react";
import { useWebSocket } from "./useWebSocket";
import type { AlertEvent, DeduplicatedAlert } from "@/api/types";

interface Options {
  enabled: boolean;
  limit?: number;
  /** @deprecated – no longer used; kept for API compat */
  interval?: number;
}

export function useAlertStream({ enabled }: Options) {
  const { alerts: wsAlerts } = useWebSocket(enabled);

  // person_id → DeduplicatedAlert
  const alertMap = useRef<Map<string, DeduplicatedAlert>>(new Map());
  // "person_id::frame_id" keys we have already counted — prevents double-
  // counting when the server re-sends the same buffer on the next tick.
  const countedKeys = useRef<Set<string>>(new Set());

  const [alerts, setAlerts] = useState<DeduplicatedAlert[]>([]);
  const [newCount, setNewCount] = useState(0);

  useEffect(() => {
    if (!enabled || !wsAlerts.length) return;

    let freshPersons = 0;

    for (const a of wsAlerts) {
      const countKey = `${a.person_id}::${a.frame_id}`;

      // Skip if we already counted this exact (person, frame) pair.
      if (countedKeys.current.has(countKey)) continue;
      countedKeys.current.add(countKey);

      const existing = alertMap.current.get(a.person_id);
      if (existing) {
        // Same person, new frame — just update metadata and bump hits.
        existing.hits++;
        existing.frame_id  = a.frame_id;
        existing.timestamp = a.timestamp;
        if (a.similarity > existing.similarity) {
          existing.similarity = a.similarity;
        }
      } else {
        // First detection of this person this session.
        alertMap.current.set(a.person_id, {
          ...a,
          hits:           1,
          first_frame_id: a.frame_id,
        });
        freshPersons++;
      }
    }

    if (freshPersons === 0 && !wsAlerts.some(
      (a) => !countedKeys.current.has(`${a.person_id}::${a.frame_id}`)
    )) {
      // Nothing new — skip re-render.
      return;
    }

    const sorted = [...alertMap.current.values()].sort(
      (a, b) => b.timestamp - a.timestamp,
    );
    setAlerts(sorted);
    if (freshPersons > 0) setNewCount((c) => c + freshPersons);
  }, [wsAlerts, enabled]);

  // Reset everything when stream is disabled.
  useEffect(() => {
    if (!enabled) {
      alertMap.current.clear();
      countedKeys.current.clear();
      setAlerts([]);
      setNewCount(0);
    }
  }, [enabled]);

  return { alerts, newCount };
}

import { useEffect, useRef, useState } from "react";
import { getAlerts } from "@/api/endpoints";
import type { AlertEvent } from "@/api/types";
import { POLL_ALERTS_MS } from "@/lib/constants";

interface Options {
  enabled: boolean;
  limit?: number;
  interval?: number;
}

const keyOf = (a: AlertEvent) => `${a.person_id}::${a.frame_id}`;

export function useAlertStream({
  enabled,
  limit = 100,
  interval = POLL_ALERTS_MS,
}: Options) {
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  const [newCount, setNewCount] = useState(0);
  const seen = useRef<Set<string>>(new Set());
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    if (!enabled) return;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const res = await getAlerts(limit);
        if (!mounted.current) return;
        const deduped: AlertEvent[] = [];
        const keys = new Set<string>();
        for (const a of res) {
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
      } catch {
        /* keep last good data */
      } finally {
        if (mounted.current) timer = setTimeout(poll, interval);
      }
    };
    poll();

    return () => {
      mounted.current = false;
      clearTimeout(timer);
    };
  }, [enabled, limit, interval]);

  return { alerts, newCount };
}

import { useEffect, useRef, useState } from "react";
import { getEvents } from "@/api/endpoints";
import type { FrameEvent } from "@/api/types";
import { POLL_EVENTS_MS } from "@/lib/constants";

interface Options {
  enabled: boolean;
  limit?: number;
  interval?: number;
}

export function useEventStream({
  enabled,
  limit = 50,
  interval = POLL_EVENTS_MS,
}: Options) {
  const [events, setEvents] = useState<FrameEvent[]>([]);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    if (!enabled) return;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const res = await getEvents(limit);
        if (mounted.current) setEvents(res);
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

  const latest = events.length ? events[events.length - 1] : null;
  return { events, latest };
}

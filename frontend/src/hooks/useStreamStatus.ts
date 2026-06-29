import { useEffect, useRef, useState } from "react";
import { getStreamStatus } from "@/api/endpoints";
import type { StreamStatus } from "@/api/types";
import { POLL_STREAM_MS } from "@/lib/constants";

interface Options {
  enabled: boolean;
  interval?: number;
}

export function useStreamStatus({ enabled, interval = POLL_STREAM_MS }: Options) {
  const [status, setStatus] = useState<StreamStatus | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    if (!enabled) return;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const res = await getStreamStatus();
        if (mounted.current) setStatus(res);
      } catch {
        if (mounted.current) setStatus(null);
      } finally {
        if (mounted.current) timer = setTimeout(poll, interval);
      }
    };
    poll();

    return () => {
      mounted.current = false;
      clearTimeout(timer);
    };
  }, [enabled, interval]);

  return { status, isLive: !!status?.running };
}

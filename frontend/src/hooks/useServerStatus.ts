import { useEffect, useRef, useState } from "react";
import { getServerStatus } from "@/api/endpoints";
import type { ServerStatus } from "@/api/types";
import { POLL_STATUS_MS } from "@/lib/constants";

export function useServerStatus(interval = POLL_STATUS_MS) {
  const [data, setData] = useState<ServerStatus | null>(null);
  const [online, setOnline] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const res = await getServerStatus();
        if (!mounted.current) return;
        setData(res);
        setOnline(true);
        setError(null);
      } catch (e) {
        if (!mounted.current) return;
        setOnline(false);
        setError(e instanceof Error ? e.message : "offline");
      } finally {
        if (mounted.current) timer = setTimeout(poll, interval);
      }
    };
    poll();

    return () => {
      mounted.current = false;
      clearTimeout(timer);
    };
  }, [interval]);

  return { data, online, error };
}

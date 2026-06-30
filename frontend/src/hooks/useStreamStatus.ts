/**
 * useStreamStatus — backed by WebSocket (falls back to polling when WS is
 * not yet connected).
 */
import { useWebSocket } from "./useWebSocket";
import type { StreamStatus } from "@/api/types";

interface Options {
  enabled: boolean;
}

export function useStreamStatus({ enabled }: Options): {
  status: StreamStatus | null;
  isLive: boolean;
  wsConnected: boolean;
} {
  const { streamStatus, connected } = useWebSocket(enabled);
  return {
    status: streamStatus,
    isLive: !!streamStatus?.running,
    wsConnected: connected,
  };
}

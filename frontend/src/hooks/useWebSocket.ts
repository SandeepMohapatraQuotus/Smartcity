/**
 * useWebSocket
 * -----------
 * Manages a single WebSocket connection to the Smart City backend.
 * Parses every incoming JSON message and distributes the three sub-payloads
 * (stream_status, latest_event, alerts) to their respective state slices.
 *
 * Features:
 *  - Exponential-backoff auto-reconnect (max 30 s)
 *  - Skips reconnect when `enabled` is false
 *  - Clean teardown on unmount / dependency change
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { AlertEvent, FrameEvent, StreamStatus } from "@/api/types";
import { WS_BASE_URL } from "@/lib/constants";

export interface WsState {
  connected: boolean;
  streamStatus: StreamStatus | null;
  latestEvent: FrameEvent | null;
  alerts: AlertEvent[];
}

const INITIAL: WsState = {
  connected: false,
  streamStatus: null,
  latestEvent: null,
  alerts: [],
};

const MAX_BACKOFF_MS = 30_000;

export function useWebSocket(enabled = true): WsState {
  const [state, setState] = useState<WsState>(INITIAL);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef(1_000);
  const mountedRef = useRef(true);

  const clearRetry = () => {
    if (retryRef.current) {
      clearTimeout(retryRef.current);
      retryRef.current = null;
    }
  };

  const connect = useCallback(() => {
    if (!mountedRef.current || !enabled) return;

    const url = `${WS_BASE_URL}/ws`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (!mountedRef.current) return;
      backoffRef.current = 1_000; // reset backoff on success
      setState((prev) => ({ ...prev, connected: true }));
    };

    ws.onmessage = (evt) => {
      if (!mountedRef.current) return;
      try {
        const msg = JSON.parse(evt.data as string) as {
          stream_status?: StreamStatus;
          latest_event?: FrameEvent | null;
          alerts?: AlertEvent[];
        };
        setState((prev) => ({
          ...prev,
          streamStatus: msg.stream_status ?? prev.streamStatus,
          latestEvent: msg.latest_event !== undefined ? msg.latest_event : prev.latestEvent,
          alerts: msg.alerts ?? prev.alerts,
        }));
      } catch {
        // malformed message — ignore
      }
    };

    ws.onerror = () => {
      // onerror is always followed by onclose; teardown happens there
    };

    ws.onclose = () => {
      if (!mountedRef.current) return;
      setState((prev) => ({ ...prev, connected: false }));
      // Exponential backoff
      clearRetry();
      retryRef.current = setTimeout(() => {
        if (!mountedRef.current) return;
        backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
        connect();
      }, backoffRef.current);
    };
  }, [enabled]);

  useEffect(() => {
    mountedRef.current = true;

    if (enabled) {
      connect();
    }

    return () => {
      mountedRef.current = false;
      clearRetry();
      if (wsRef.current) {
        // Prevent the onclose handler from scheduling a reconnect
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
      setState(INITIAL);
    };
  }, [enabled, connect]);

  return state;
}

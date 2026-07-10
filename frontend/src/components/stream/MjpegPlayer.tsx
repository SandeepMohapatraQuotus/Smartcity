import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { VideoOff, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { MJPEG_STREAM_URL } from "@/lib/constants";

interface MjpegPlayerProps {
  source: string | null;
  live: boolean;
  cameraId?: string;
  className?: string;
}

/**
 * MjpegPlayer
 * -----------
 * Renders the MJPEG stream from GET /stream/mjpeg via a plain <img> tag.
 * The server emits a continuous multipart/x-mixed-replace response so the
 * browser updates the frame in place without any JS polling.
 *
 * Auto-retry: on error the key increments, which unmounts + remounts the <img>
 * after a short delay, triggering a fresh HTTP request to the server.
 */
export function MjpegPlayer({ source, live, cameraId, className }: MjpegPlayerProps) {
  const [imgKey, setImgKey] = useState(0);          // bumped to force <img> remount on error
  const [loading, setLoading] = useState(false);
  const retryTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryDelay = useRef(1_000);

  // Build the stream URL — always point at /stream/mjpeg.
  // Add ?t= cache-buster so the browser never serves a stale response.
  // The `source` query param is forwarded for compatibility but the server
  // uses whatever was supplied to POST /stream/start.
  const streamSrc = live
    ? `${MJPEG_STREAM_URL}?source=${encodeURIComponent(source ?? "")}&t=${imgKey}`
    : null;

  // Reset retry state when live status changes
  useEffect(() => {
    if (!live) {
      retryDelay.current = 1_000;
      if (retryTimer.current) {
        clearTimeout(retryTimer.current);
        retryTimer.current = null;
      }
      setLoading(false);
    } else {
      setLoading(true);
    }
  }, [live]);

  const handleLoad = () => {
    setLoading(false);
    retryDelay.current = 1_000; // reset backoff on success
  };

  const handleError = () => {
    if (!live) return;
    setLoading(false);
    retryTimer.current = setTimeout(() => {
      retryDelay.current = Math.min(retryDelay.current * 2, 10_000);
      setImgKey((k) => k + 1); // remount <img> → fresh request
      setLoading(true);
    }, retryDelay.current);
  };

  return (
    <div
      className={cn(
        "relative aspect-video w-full overflow-hidden rounded-xl border border-surface-border bg-black",
        className,
      )}
    >
      {streamSrc ? (
        <img
          key={imgKey}
          src={streamSrc}
          alt="live annotated stream"
          onLoad={handleLoad}
          onError={handleError}
          className="h-full w-full object-contain"
        />
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
          <VideoOff className="h-10 w-10" />
          <span className="text-sm">No stream connected</span>
        </div>
      )}

      {/* Connecting / retry overlay */}
      {live && loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/60 text-muted-foreground">
          <Loader2 className="h-8 w-8 animate-spin" />
          <span className="text-sm">Connecting to stream…</span>
        </div>
      )}

      {/* LIVE badge */}
      {live && (
        <motion.div
          className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-surface-deep/80 px-3 py-1 text-xs font-semibold text-status-alert backdrop-blur"
          animate={{ opacity: [1, 0.4, 1] }}
          transition={{ duration: 1.4, repeat: Infinity }}
        >
          <span className="h-2 w-2 rounded-full bg-status-alert" /> REC
        </motion.div>
      )}

      {cameraId && (
        <span className="absolute right-3 top-3 rounded-full bg-surface-deep/80 px-3 py-1 font-mono text-xs text-brand backdrop-blur">
          {cameraId}
        </span>
      )}
    </div>
  );
}

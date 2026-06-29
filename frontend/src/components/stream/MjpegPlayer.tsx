import { useState } from "react";
import { motion } from "framer-motion";
import { VideoOff } from "lucide-react";
import { cn } from "@/lib/utils";
import { MJPEG_STREAM_URL } from "@/lib/constants";

interface MjpegPlayerProps {
  source: string | null;
  live: boolean;
  cameraId?: string;
  className?: string;
}

export function MjpegPlayer({ source, live, cameraId, className }: MjpegPlayerProps) {
  const [errored, setErrored] = useState(false);
  const streamSrc =
    live && source
      ? `${MJPEG_STREAM_URL}?source=${encodeURIComponent(source)}&t=${live ? "1" : "0"}`
      : null;

  return (
    <div
      className={cn(
        "relative aspect-video w-full overflow-hidden rounded-xl border border-surface-border bg-black",
        className,
      )}
    >
      {streamSrc && !errored ? (
        <img
          src={streamSrc}
          alt="live stream"
          onError={() => setErrored(true)}
          className="h-full w-full object-contain"
        />
      ) : (
        <div className="flex h-full flex-col items-center justify-center gap-2 text-muted-foreground">
          <VideoOff className="h-10 w-10" />
          <span className="text-sm">
            {errored ? "Stream unavailable" : "No stream connected"}
          </span>
        </div>
      )}

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

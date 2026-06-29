import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

type Status = "live" | "offline" | "error" | "loading";

const styles: Record<Status, { dot: string; text: string; label: string }> = {
  live: { dot: "bg-status-live", text: "text-status-live", label: "LIVE" },
  offline: { dot: "bg-muted-foreground", text: "text-muted-foreground", label: "OFFLINE" },
  error: { dot: "bg-status-alert", text: "text-status-alert", label: "ERROR" },
  loading: { dot: "bg-status-warn", text: "text-status-warn", label: "LOADING" },
};

interface StatusBadgeProps {
  status: Status;
  label?: string;
  className?: string;
}

export function StatusBadge({ status, label, className }: StatusBadgeProps) {
  const s = styles[status];
  const animated = status === "live" || status === "loading";

  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border border-surface-border bg-surface-deep px-2.5 py-1 text-[11px] font-semibold tracking-wider",
        s.text,
        className,
      )}
    >
      <span className="relative flex h-2 w-2">
        {animated && (
          <motion.span
            className={cn("absolute inline-flex h-full w-full rounded-full opacity-75", s.dot)}
            animate={{ scale: [1, 2.2, 1], opacity: [0.7, 0, 0.7] }}
            transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }}
          />
        )}
        <span className={cn("relative inline-flex h-2 w-2 rounded-full", s.dot)} />
      </span>
      {label ?? s.label}
    </span>
  );
}

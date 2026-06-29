import { useState } from "react";
import { ChevronRight } from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { cn } from "@/lib/utils";

function Primitive({ value }: { value: unknown }) {
  if (value === null) return <span className="text-muted-foreground">null</span>;
  if (typeof value === "string")
    return <span className="text-status-live">&quot;{value}&quot;</span>;
  if (typeof value === "number")
    return <span className="text-brand">{value}</span>;
  if (typeof value === "boolean")
    return <span className="text-status-night">{String(value)}</span>;
  return <span>{String(value)}</span>;
}

function Node({ name, value, depth }: { name?: string; value: unknown; depth: number }) {
  const isObject = value !== null && typeof value === "object";
  const [open, setOpen] = useState(depth < 1);

  if (!isObject) {
    return (
      <div className="flex gap-2 py-0.5" style={{ paddingLeft: depth * 14 }}>
        {name !== undefined && <span className="text-foreground/80">{name}:</span>}
        <Primitive value={value} />
      </div>
    );
  }

  const entries = Array.isArray(value)
    ? value.map((v, i) => [String(i), v] as const)
    : Object.entries(value as Record<string, unknown>);
  const bracket = Array.isArray(value) ? ["[", "]"] : ["{", "}"];

  return (
    <div style={{ paddingLeft: depth * 14 }}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 py-0.5 text-left hover:text-brand"
      >
        <ChevronRight
          className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-90")}
        />
        {name !== undefined && <span className="text-foreground/80">{name}:</span>}
        <span className="text-muted-foreground">
          {bracket[0]}
          {!open && ` ${entries.length} `}
          {!open && bracket[1]}
        </span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18 }}
            className="overflow-hidden"
          >
            {entries.map(([k, v]) => (
              <Node key={k} name={k} value={v} depth={depth + 1} />
            ))}
            <div
              className="text-muted-foreground"
              style={{ paddingLeft: depth * 14 }}
            >
              {bracket[1]}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function JsonViewer({ data, className }: { data: unknown; className?: string }) {
  return (
    <div
      className={cn(
        "max-h-[28rem] overflow-auto rounded-lg border border-surface-border bg-surface-deep p-4 font-mono text-xs leading-relaxed",
        className,
      )}
    >
      <Node value={data} depth={0} />
    </div>
  );
}

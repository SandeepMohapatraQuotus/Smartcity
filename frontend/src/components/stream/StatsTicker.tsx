import { useEffect } from "react";
import { motion, useSpring, useTransform } from "framer-motion";
import { cn } from "@/lib/utils";

function Counter({ value }: { value: number }) {
  const spring = useSpring(0, { stiffness: 80, damping: 20 });
  const rounded = useTransform(spring, (v) => Math.round(v).toLocaleString());
  useEffect(() => {
    spring.set(value);
  }, [value, spring]);
  return <motion.span>{rounded}</motion.span>;
}

export interface StatItem {
  label: string;
  value: number;
  accent?: string;
}

export function StatsTicker({
  stats,
  className,
}: {
  stats: StatItem[];
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-surface-border bg-surface-border sm:grid-cols-3 lg:grid-cols-5",
        className,
      )}
    >
      {stats.map((s) => (
        <div key={s.label} className="bg-card px-4 py-5 text-center">
          <div className={cn("font-mono text-3xl font-bold", s.accent ?? "text-brand")}>
            <Counter value={s.value} />
          </div>
          <div className="mt-1 text-xs uppercase tracking-wider text-muted-foreground">
            {s.label}
          </div>
        </div>
      ))}
    </div>
  );
}

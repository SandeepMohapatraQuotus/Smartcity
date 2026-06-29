import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export function PageWrapper({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={cn("mx-auto w-full max-w-6xl space-y-6", className)}
    >
      {children}
    </motion.div>
  );
}

export function PageHeader({
  title,
  endpoint,
  description,
  action,
}: {
  title: string;
  endpoint?: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-4">
      <div className="min-w-0">
        <h1 className="truncate font-mono text-2xl font-bold tracking-tight">
          {title}
        </h1>
        {description && (
          <p className="mt-1 text-sm text-muted-foreground">{description}</p>
        )}
        {endpoint && (
          <code className="mt-2 inline-block rounded bg-surface-deep px-2 py-1 font-mono text-xs text-brand">
            {endpoint}
          </code>
        )}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

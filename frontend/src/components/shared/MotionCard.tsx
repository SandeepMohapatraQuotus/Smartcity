import { motion, type HTMLMotionProps } from "framer-motion";
import { cn } from "@/lib/utils";

interface MotionCardProps extends Omit<HTMLMotionProps<"div">, "title"> {
  title?: string;
  badge?: React.ReactNode;
  children?: React.ReactNode;
}

export function MotionCard({
  title,
  badge,
  className,
  children,
  ...props
}: MotionCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className={cn(
        "rounded-xl border border-surface-border bg-card p-5 shadow-sm",
        className,
      )}
      {...props}
    >
      {(title || badge) && (
        <div className="mb-4 flex items-center justify-between gap-2">
          {title && (
            <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              {title}
            </h3>
          )}
          {badge}
        </div>
      )}
      {children}
    </motion.div>
  );
}

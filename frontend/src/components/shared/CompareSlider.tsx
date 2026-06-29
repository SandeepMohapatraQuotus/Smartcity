import { useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface CompareSliderProps {
  before: string;
  after: string;
  beforeLabel?: string;
  afterLabel?: string;
  className?: string;
}

export function CompareSlider({
  before,
  after,
  beforeLabel = "Before",
  afterLabel = "After",
  className,
}: CompareSliderProps) {
  const [pos, setPos] = useState(50);
  const ref = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const move = (clientX: number) => {
    const rect = ref.current?.getBoundingClientRect();
    if (!rect) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.min(100, Math.max(0, pct)));
  };

  return (
    <div
      ref={ref}
      className={cn(
        "relative aspect-video w-full select-none overflow-hidden rounded-lg border border-surface-border",
        className,
      )}
      onMouseMove={(e) => dragging.current && move(e.clientX)}
      onMouseDown={(e) => {
        dragging.current = true;
        move(e.clientX);
      }}
      onMouseUp={() => (dragging.current = false)}
      onMouseLeave={() => (dragging.current = false)}
      onTouchMove={(e) => move(e.touches[0].clientX)}
    >
      <img src={after} alt={afterLabel} className="absolute inset-0 h-full w-full object-contain" />
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      >
        <img src={before} alt={beforeLabel} className="absolute inset-0 h-full w-full object-contain" />
      </div>

      <span className="absolute left-2 top-2 rounded bg-surface-deep/80 px-2 py-0.5 text-[11px] font-semibold text-foreground">
        {beforeLabel}
      </span>
      <span className="absolute right-2 top-2 rounded bg-surface-deep/80 px-2 py-0.5 text-[11px] font-semibold text-foreground">
        {afterLabel}
      </span>

      <div
        className="absolute inset-y-0 w-0.5 bg-brand"
        style={{ left: `${pos}%` }}
      >
        <div className="absolute top-1/2 left-1/2 grid h-7 w-7 -translate-x-1/2 -translate-y-1/2 place-items-center rounded-full border border-brand bg-surface-deep text-brand">
          ⇄
        </div>
      </div>
    </div>
  );
}

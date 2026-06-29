import { useEffect, useState } from "react";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Separator } from "@/components/ui/separator";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { useServerStatus } from "@/hooks/useServerStatus";
import { DEFAULT_CAMERA_ID } from "@/lib/constants";

export function TopBar() {
  const { online, data } = useServerStatus();
  const [clock, setClock] = useState("");

  useEffect(() => {
    const tick = () =>
      setClock(new Date().toLocaleTimeString("en-GB", { hour12: false }));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="sticky top-0 z-10 flex h-14 items-center gap-3 border-b border-surface-border bg-surface-deep/80 px-3 backdrop-blur">
      <SidebarTrigger />
      <Separator orientation="vertical" className="h-6" />
      <span className="font-mono text-sm font-bold tracking-tight">
        Smart City Monitor
      </span>
      <div className="ml-auto flex items-center gap-3">
        <StatusBadge
          status={online ? (data?.status === "ready" ? "live" : "loading") : "offline"}
        />
        <span className="hidden font-mono text-xs text-muted-foreground sm:inline">
          {DEFAULT_CAMERA_ID}
        </span>
        <span className="font-mono text-sm tabular-nums text-brand">{clock}</span>
      </div>
    </header>
  );
}

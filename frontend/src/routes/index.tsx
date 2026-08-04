import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Sun, Moon, Car, User, ScanText, ScanFace, Siren, Wifi, WifiOff } from "lucide-react";
import { toast } from "sonner";
import { PageWrapper } from "@/components/layout/PageHeader";
import { MotionCard } from "@/components/shared/MotionCard";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { MjpegPlayer } from "@/components/stream/MjpegPlayer";
import { StatsTicker } from "@/components/stream/StatsTicker";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { useStreamStatus } from "@/hooks/useStreamStatus";
import { useEventStream } from "@/hooks/useEventStream";
import { useAlertStream } from "@/hooks/useAlertStream";
import { startStream, stopStream } from "@/api/endpoints";
import { DEFAULT_CAMERA_ID } from "@/lib/constants";
import { cn, formatTimestamp } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Dashboard — Smart City Monitor" },
      {
        name: "description",
        content: "Live annotated camera stream with real-time vehicle, person, plate and face stats.",
      },
    ],
  }),
  component: Dashboard,
});

function StatRow({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Car;
  label: string;
  value: number;
}) {
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="flex items-center gap-2 text-sm text-muted-foreground">
        <Icon className="h-4 w-4" /> {label}
      </span>
      <span className="font-mono text-base font-bold text-foreground">{value}</span>
    </div>
  );
}

function Dashboard() {
  const [source, setSource] = useState("");
  const [busy, setBusy] = useState(false);

  const { status, isLive, wsConnected } = useStreamStatus({ enabled: true });
  const { latest } = useEventStream({ enabled: isLive, limit: 1 });
  const { alerts } = useAlertStream({ enabled: isLive });

  const totals = useMemo(() => {
    return {
      frames: status?.frames_processed ?? 0,
      vehicles: latest?.vehicles?.total ?? 0,
      persons: latest?.persons?.person_count ?? 0,
      plates: latest?.plates?.plate_count ?? 0,
      alerts: alerts.length,
    };
  }, [status, latest, alerts]);

  const dn = latest?.day_night;
  const isDay = dn?.label !== "night";

  const handleConnect = async () => {
    if (!source.trim()) {
      toast.error("Enter a stream source first");
      return;
    }
    setBusy(true);
    try {
      await startStream(source.trim(), DEFAULT_CAMERA_ID);
      toast.success("Stream started");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to start stream");
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    setBusy(true);
    try {
      await stopStream();
      toast.success("Stream stopped");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to stop stream");
    } finally {
      setBusy(false);
    }
  };

  return (
    <PageWrapper className="max-w-7xl">
      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="space-y-4">
          <MjpegPlayer
            source={status?.source ?? source}
            live={isLive}
            cameraId={status?.camera_id ?? DEFAULT_CAMERA_ID}
          />
          <MotionCard>
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
              <Input
                placeholder="rtsp://… or video URL"
                value={source}
                onChange={(e) => setSource(e.target.value)}
                className="font-mono"
              />
              <div className="flex items-center gap-2">
                <Button onClick={handleConnect} disabled={busy}>
                  Connect
                </Button>
                <Button variant="secondary" onClick={handleStop} disabled={busy}>
                  Stop
                </Button>
                <StatusBadge status={isLive ? "live" : "offline"} />
                {/* WebSocket connection indicator */}
                <span
                  title={wsConnected ? "WebSocket connected" : "WebSocket disconnected"}
                  className={cn(
                    "flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold",
                    wsConnected
                      ? "bg-emerald-500/15 text-emerald-400"
                      : "bg-red-500/15 text-red-400",
                  )}
                >
                  {wsConnected ? (
                    <Wifi className="h-3.5 w-3.5" />
                  ) : (
                    <WifiOff className="h-3.5 w-3.5" />
                  )}
                  WS
                </span>
              </div>
            </div>
            {status?.error && (
              <p className="mt-2 text-xs text-status-alert">{status.error}</p>
            )}
          </MotionCard>
        </div>

        <div className="space-y-4">
          <MotionCard title="Scene Condition">
            <div
              className={cn(
                "flex items-center gap-3 rounded-lg p-3",
                isDay ? "glow-day bg-status-day/10" : "glow-night bg-status-night/10",
              )}
            >
              {isDay ? (
                <Sun className="h-8 w-8 text-status-day" />
              ) : (
                <Moon className="h-8 w-8 text-status-night" />
              )}
              <div className="flex-1">
                <div className="font-mono text-lg font-bold uppercase">
                  {dn?.label ?? "—"}
                </div>
                <Progress
                  value={Math.round((dn?.confidence ?? 0) * 100)}
                  className="mt-1 h-1.5"
                />
              </div>
              <span className="font-mono text-sm">
                {Math.round((dn?.confidence ?? 0) * 100)}%
              </span>
            </div>
          </MotionCard>

          <MotionCard title="Current Frame">
            <StatRow icon={Car} label="Vehicles" value={totals.vehicles} />
            <StatRow icon={User} label="Persons" value={totals.persons} />
            <StatRow icon={ScanText} label="Plates" value={totals.plates} />
            <StatRow icon={ScanFace} label="Faces" value={latest?.faces?.length ?? 0} />
          </MotionCard>

          <MotionCard title="Alerts">
            <div className="space-y-2">
              <AnimatePresence initial={false}>
                {alerts.slice(0, 6).map((a) => (
                  <motion.div
                    key={a.person_id}
                    initial={{ opacity: 0, y: -10, scale: 0.96 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.96 }}
                    className="glow-alert rounded-lg border border-status-alert/40 bg-status-alert/5 p-3"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2.5 min-w-0">
                        {/* Person photo or icon */}
                        {a.image_url ? (
                          <img
                            src={a.image_url}
                            alt={a.name}
                            className="h-8 w-8 shrink-0 rounded-full object-cover ring-1 ring-status-alert/50"
                          />
                        ) : (
                          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-status-alert/20">
                            <Siren className="h-4 w-4 text-status-alert" />
                          </div>
                        )}
                        <span className="truncate text-sm font-semibold text-status-alert">
                          {a.name}
                        </span>
                      </div>
                      {a.hits > 1 && (
                        <span className="shrink-0 rounded-full bg-status-alert/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-status-alert">
                          ×{a.hits}
                        </span>
                      )}
                    </div>
                    <div className="mt-1 font-mono text-xs text-muted-foreground">
                      sim {Math.round(a.similarity * 100)}% · {a.frame_id} ·{" "}
                      {formatTimestamp(a.timestamp)}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
              {alerts.length === 0 && (
                <p className="py-4 text-center text-sm text-muted-foreground">
                  No active alerts
                </p>
              )}
            </div>
          </MotionCard>
        </div>
      </div>

      <StatsTicker
        stats={[
          { label: "Frames", value: totals.frames },
          { label: "Vehicles", value: totals.vehicles, accent: "text-status-day" },
          { label: "Persons", value: totals.persons, accent: "text-status-live" },
          { label: "Plates", value: totals.plates, accent: "text-brand" },
          { label: "Alerts", value: totals.alerts, accent: "text-status-alert" },
        ]}
      />
    </PageWrapper>
  );
}

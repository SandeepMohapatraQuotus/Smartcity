import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { RefreshCw, Trash2, Sun, Moon, Siren } from "lucide-react";
import { toast } from "sonner";
import { PageWrapper, PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { JsonViewer } from "@/components/shared/JsonViewer";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { getEvents, getAlerts, clearBuffers } from "@/api/endpoints";
import type { FrameEvent, AlertEvent, DeduplicatedAlert } from "@/api/types";
import { formatTimestamp } from "@/lib/utils";

export const Route = createFileRoute("/events")({
  head: () => ({
    meta: [
      { title: "Events & Alerts — Smart City Monitor" },
      { name: "description", content: "Inspect and clear the live event and alert ring buffers." },
    ],
  }),
  component: Events,
});

// ── Alert deduplication ────────────────────────────────────────────
// Collapse the raw alert list (one entry per frame hit) into one entry
// per unique person_id, tracking hit count and first/latest frames.
function dedupeAlerts(raw: AlertEvent[]): DeduplicatedAlert[] {
  const map = new Map<string, DeduplicatedAlert>();
  for (const a of raw) {
    const existing = map.get(a.person_id);
    if (existing) {
      existing.hits++;
      // Keep the latest frame/timestamp.
      existing.frame_id  = a.frame_id;
      existing.timestamp = a.timestamp;
      if (a.similarity > existing.similarity) existing.similarity = a.similarity;
    } else {
      map.set(a.person_id, { ...a, hits: 1, first_frame_id: a.frame_id });
    }
  }
  // Sort by most-recently seen first.
  return [...map.values()].sort((a, b) => b.timestamp - a.timestamp);
}

function ClearButton({ onCleared }: { onCleared: () => void }) {
  return (
    <AlertDialog>
      <AlertDialogTrigger asChild>
        <Button variant="secondary" size="sm">
          <Trash2 className="mr-2 h-4 w-4" /> Clear Buffers
        </Button>
      </AlertDialogTrigger>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Clear all buffers?</AlertDialogTitle>
          <AlertDialogDescription>
            This clears both the event and alert ring buffers on the server.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>Cancel</AlertDialogCancel>
          <AlertDialogAction
            onClick={async () => {
              try {
                await clearBuffers();
                toast.success("Buffers cleared");
                onCleared();
              } catch (e) {
                toast.error(e instanceof Error ? e.message : "Failed to clear");
              }
            }}
          >
            Clear
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

function EventRow({ ev, index }: { ev: FrameEvent; index: number }) {
  const [open, setOpen] = useState(false);
  const isDay = ev.day_night?.label !== "night";
  return (
    <motion.div
      initial={{ opacity: 0, x: -12 }}
      animate={{ opacity: 1, x: 0, transition: { delay: index * 0.03 } }}
      className="border-b border-surface-border py-3"
    >
      <button onClick={() => setOpen((o) => !o)} className="w-full text-left">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="font-mono text-brand">{ev.frame_id}</span>
          <span className="text-muted-foreground">· {ev.camera_id} ·</span>
          <span className="font-mono text-muted-foreground">
            {formatTimestamp(ev.timestamp)}
          </span>
          {isDay ? (
            <Sun className="h-4 w-4 text-status-day" />
          ) : (
            <Moon className="h-4 w-4 text-status-night" />
          )}
        </div>
        <div className="mt-1 flex gap-4 font-mono text-xs text-muted-foreground">
          <span>🚗 {ev.vehicles?.total ?? 0}</span>
          <span>🧍 {ev.persons?.person_count ?? 0}</span>
          <span>🪪 {ev.plates?.plate_count ?? 0}</span>
          <span>👤 {ev.faces?.length ?? 0}</span>
          <span>🚨 {ev.alerts?.length ?? 0}</span>
        </div>
      </button>
      {open && (
        <div className="mt-2">
          <JsonViewer data={ev} />
        </div>
      )}
    </motion.div>
  );
}

function EventTab() {
  const [events, setEvents] = useState<FrameEvent[]>([]);
  const [limit, setLimit] = useState(50);
  const [auto, setAuto] = useState(false);

  const load = async () => {
    try {
      setEvents(await getEvents(limit));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to load events");
    }
  };

  useEffect(() => {
    load();
  }, [limit]);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(load, 2000);
    return () => clearInterval(id);
  }, [auto, limit]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-muted-foreground">{events.length} events</span>
        <div className="ml-auto flex flex-wrap items-center gap-3">
          <Select value={String(limit)} onValueChange={(v) => setLimit(Number(v))}>
            <SelectTrigger className="w-24">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[10, 50, 100, 500].map((n) => (
                <SelectItem key={n} value={String(n)}>
                  {n}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="flex items-center gap-2">
            <Switch id="auto" checked={auto} onCheckedChange={setAuto} />
            <Label htmlFor="auto" className="text-sm">
              Auto
            </Label>
          </div>
          <Button variant="secondary" size="sm" onClick={load}>
            <RefreshCw className="mr-2 h-4 w-4" /> Refresh
          </Button>
          <ClearButton onCleared={load} />
        </div>
      </div>
      <ScrollArea className="h-[60vh] rounded-xl border border-surface-border bg-card px-4">
        {events.length === 0 ? (
          <p className="py-12 text-center text-sm text-muted-foreground">
            No events in buffer.
          </p>
        ) : (
          [...events].reverse().map((ev, i) => (
            <EventRow key={ev.frame_id + i} ev={ev} index={i} />
          ))
        )}
      </ScrollArea>
    </div>
  );
}

function AlertTab() {
  const [alerts, setAlerts] = useState<DeduplicatedAlert[]>([]);
  const [rawCount, setRawCount] = useState(0);

  const load = async () => {
    try {
      const raw = await getAlerts(100);
      setRawCount(raw.length);
      setAlerts(dedupeAlerts(raw));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to load alerts");
    }
  };

  useEffect(() => {
    load();
  }, []);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <span className="text-sm text-muted-foreground">
          {alerts.length} unique person{alerts.length === 1 ? "" : "s"}
          {rawCount > alerts.length && (
            <span className="ml-1 text-muted-foreground/60">({rawCount} total detections)</span>
          )}
        </span>
        <div className="ml-auto flex gap-2">
          <Button variant="secondary" size="sm" onClick={load}>
            <RefreshCw className="mr-2 h-4 w-4" /> Refresh
          </Button>
          <ClearButton onCleared={load} />
        </div>
      </div>
      <div className="space-y-3">
        {alerts.length === 0 ? (
          <p className="py-12 text-center text-sm text-muted-foreground">No alerts.</p>
        ) : (
          alerts.map((a, i) => (
            <motion.div
              key={a.person_id}
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0, transition: { delay: i * 0.04 } }}
              className="glow-alert rounded-xl border border-status-alert/40 bg-status-alert/5 p-4"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {/* Person photo or icon */}
                  {a.image_url ? (
                    <img
                      src={a.image_url}
                      alt={a.name}
                      className="h-10 w-10 rounded-full object-cover ring-2 ring-status-alert/40"
                    />
                  ) : (
                    <div className="flex h-10 w-10 items-center justify-center rounded-full bg-status-alert/20">
                      <Siren className="h-5 w-5 text-status-alert" />
                    </div>
                  )}
                  <div>
                    <div className="flex items-center gap-2 font-semibold text-status-alert">
                      {a.name}
                      <span className="font-mono text-xs text-muted-foreground">({a.person_id})</span>
                    </div>
                    <div className="font-mono text-xs text-muted-foreground">
                      Best sim: {(a.similarity * 100).toFixed(1)}%
                      {a.method && <span className="ml-1">via {a.method}</span>}
                    </div>
                  </div>
                </div>
                {/* Hit counter badge */}
                {a.hits > 1 && (
                  <span className="rounded-full bg-status-alert/20 px-2 py-0.5 font-mono text-xs font-bold text-status-alert">
                    ×{a.hits} hits
                  </span>
                )}
              </div>
              <div className="mt-1.5 font-mono text-xs text-muted-foreground/60">
                {a.hits > 1
                  ? <>First: {a.first_frame_id} · Latest: {a.frame_id}</>
                  : a.frame_id
                }
                {" · "}{a.camera_id} · {formatTimestamp(a.timestamp)}
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
}

function Events() {
  return (
    <PageWrapper>
      <PageHeader title="Events & Alerts" endpoint="GET /events · /alerts · DELETE /events" />
      <Tabs defaultValue="events">
        <TabsList>
          <TabsTrigger value="events">Event Buffer</TabsTrigger>
          <TabsTrigger value="alerts">Alert Feed</TabsTrigger>
        </TabsList>
        <TabsContent value="events">
          <EventTab />
        </TabsContent>
        <TabsContent value="alerts">
          <AlertTab />
        </TabsContent>
      </Tabs>
    </PageWrapper>
  );
}

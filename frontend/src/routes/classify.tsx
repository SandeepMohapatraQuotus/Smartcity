import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Sun, Moon } from "lucide-react";
import { toast } from "sonner";
import { PageWrapper, PageHeader } from "@/components/layout/PageHeader";
import { MotionCard } from "@/components/shared/MotionCard";
import { ImageUploader } from "@/components/shared/ImageUploader";
import { CompareSlider } from "@/components/shared/CompareSlider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { classifyDayNight, enhanceFrame } from "@/api/endpoints";
import type { DayNightResult } from "@/api/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/classify")({
  head: () => ({
    meta: [
      { title: "Day / Night & Enhancement — Smart City Monitor" },
      { name: "description", content: "Classify day vs night and enhance dark frames." },
    ],
  }),
  component: Classify,
});

function DayNightTab() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DayNightResult | null>(null);

  const run = async () => {
    if (!file) return toast.error("Upload an image first");
    setLoading(true);
    setResult(null);
    try {
      setResult(await classifyDayNight(file));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Classification failed");
    } finally {
      setLoading(false);
    }
  };

  const isDay = result?.label !== "night";

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <MotionCard title="Upload Image">
        <ImageUploader onFile={setFile} />
        <Button className="mt-4 w-full" onClick={run} disabled={loading}>
          {loading ? "Classifying…" : "Classify →"}
        </Button>
      </MotionCard>
      <MotionCard title="Result">
        {result ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            className="space-y-4"
          >
            <div
              className={cn(
                "flex items-center gap-4 rounded-xl p-5",
                isDay ? "glow-day bg-status-day/10" : "glow-night bg-status-night/10",
              )}
            >
              {isDay ? (
                <Sun className="h-12 w-12 text-status-day" />
              ) : (
                <Moon className="h-12 w-12 text-status-night" />
              )}
              <span className="font-mono text-3xl font-bold uppercase">
                {result.label}
              </span>
            </div>
            <div>
              <div className="mb-1 flex justify-between text-sm">
                <span className="text-muted-foreground">Confidence</span>
                <span className="font-mono">{Math.round(result.confidence * 100)}%</span>
              </div>
              <Progress value={Math.round(result.confidence * 100)} />
            </div>
            <div className="flex gap-2 text-xs">
              <Badge variant="secondary">method: {result.method}</Badge>
              <Badge variant="secondary">
                enhance: {String(result.route_to_enhancement)}
              </Badge>
            </div>
          </motion.div>
        ) : (
          <p className="py-12 text-center text-sm text-muted-foreground">
            No result yet.
          </p>
        )}
      </MotionCard>
    </div>
  );
}

function EnhanceTab() {
  const [file, setFile] = useState<File | null>(null);
  const [before, setBefore] = useState<string | null>(null);
  const [after, setAfter] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(
    () => () => {
      if (after) URL.revokeObjectURL(after);
    },
    [after],
  );

  const onFile = (f: File | null) => {
    setFile(f);
    setBefore(f ? URL.createObjectURL(f) : null);
    setAfter(null);
  };

  const run = async () => {
    if (!file) return toast.error("Upload an image first");
    setLoading(true);
    try {
      const blob = await enhanceFrame(file);
      setAfter(URL.createObjectURL(blob));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Enhancement failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
      <MotionCard title="Upload Dark Image">
        <ImageUploader onFile={onFile} />
        <Button className="mt-4 w-full" onClick={run} disabled={loading}>
          {loading ? "Enhancing…" : "Enhance →"}
        </Button>
      </MotionCard>
      <MotionCard title="Before / After">
        {before && after ? (
          <div className="space-y-3">
            <CompareSlider before={before} after={after} afterLabel="Enhanced" />
            <Badge variant="secondary">zero_dce++</Badge>
          </div>
        ) : (
          <p className="py-12 text-center text-sm text-muted-foreground">
            Drag the divider to compare once enhanced.
          </p>
        )}
      </MotionCard>
    </div>
  );
}

function Classify() {
  return (
    <PageWrapper>
      <PageHeader title="Classifiers" endpoint="POST /classify/day-night · /enhance/frame" />
      <Tabs defaultValue="daynight">
        <TabsList>
          <TabsTrigger value="daynight">Day / Night</TabsTrigger>
          <TabsTrigger value="enhance">Enhance Frame</TabsTrigger>
        </TabsList>
        <TabsContent value="daynight">
          <DayNightTab />
        </TabsContent>
        <TabsContent value="enhance">
          <EnhanceTab />
        </TabsContent>
      </Tabs>
    </PageWrapper>
  );
}

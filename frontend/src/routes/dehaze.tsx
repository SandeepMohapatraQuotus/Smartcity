import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import { PageWrapper, PageHeader } from "@/components/layout/PageHeader";
import { MotionCard } from "@/components/shared/MotionCard";
import { ImageUploader } from "@/components/shared/ImageUploader";
import { AnnotatedImage } from "@/components/shared/AnnotatedImage";
import { CompareSlider } from "@/components/shared/CompareSlider";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { dehazeFrame, dehazeCompare } from "@/api/endpoints";

export const Route = createFileRoute("/dehaze")({
  head: () => ({
    meta: [
      { title: "Dehazing — Smart City Monitor" },
      { name: "description", content: "Remove haze, fog and smoke from camera frames." },
    ],
  }),
  component: Dehaze,
});

function DehazeTab() {
  const [file, setFile] = useState<File | null>(null);
  const [strength, setStrength] = useState(0.85);
  const [loading, setLoading] = useState(false);
  const [blob, setBlob] = useState<Blob | null>(null);

  const run = async () => {
    if (!file) return toast.error("Upload an image first");
    setLoading(true);
    setBlob(null);
    try {
      setBlob(await dehazeFrame(file, strength));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Dehaze failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <MotionCard title="Input">
        <ImageUploader onFile={setFile} />
        <div className="mt-4 space-y-2">
          <Label className="flex justify-between">
            <span>Strength</span>
            <span className="font-mono text-brand">{strength.toFixed(2)}</span>
          </Label>
          <Slider
            value={[strength]}
            min={0.5}
            max={1}
            step={0.05}
            onValueChange={([v]) => setStrength(v)}
          />
        </div>
        <Button className="mt-4 w-full" onClick={run} disabled={loading}>
          {loading ? "Dehazing…" : "Dehaze →"}
        </Button>
      </MotionCard>
      <MotionCard title="Output">
        <AnnotatedImage blob={blob} filename="dehazed.jpg" />
      </MotionCard>
    </div>
  );
}

function CompareTab() {
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
      const blob = await dehazeCompare(file);
      setAfter(URL.createObjectURL(blob));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Compare failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
      <MotionCard title="Input">
        <ImageUploader onFile={onFile} />
        <Button className="mt-4 w-full" onClick={run} disabled={loading}>
          {loading ? "Processing…" : "Compare →"}
        </Button>
        <Badge variant="secondary" className="mt-3">
          dcp
        </Badge>
      </MotionCard>
      <MotionCard title="Comparison">
        {before && after ? (
          <CompareSlider before={before} after={after} afterLabel="Dehazed" />
        ) : (
          <p className="py-12 text-center text-sm text-muted-foreground">
            Run compare to reveal the before/after slider.
          </p>
        )}
      </MotionCard>
    </div>
  );
}

function Dehaze() {
  return (
    <PageWrapper>
      <PageHeader title="Dehazing" endpoint="POST /dehaze/frame · /dehaze/frame/compare" />
      <Tabs defaultValue="dehaze">
        <TabsList>
          <TabsTrigger value="dehaze">Dehaze</TabsTrigger>
          <TabsTrigger value="compare">Compare</TabsTrigger>
        </TabsList>
        <TabsContent value="dehaze">
          <DehazeTab />
        </TabsContent>
        <TabsContent value="compare">
          <CompareTab />
        </TabsContent>
      </Tabs>
    </PageWrapper>
  );
}

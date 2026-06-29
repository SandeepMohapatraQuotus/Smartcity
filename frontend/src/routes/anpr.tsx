import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { motion } from "framer-motion";
import { Copy, Check } from "lucide-react";
import { toast } from "sonner";
import { PageWrapper, PageHeader } from "@/components/layout/PageHeader";
import { MotionCard } from "@/components/shared/MotionCard";
import { ImageUploader } from "@/components/shared/ImageUploader";
import { AnnotatedImage } from "@/components/shared/AnnotatedImage";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { readPlates, readPlatesAnnotated } from "@/api/endpoints";
import type { ANPRResult } from "@/api/types";

export const Route = createFileRoute("/anpr")({
  head: () => ({
    meta: [
      { title: "ANPR — Smart City Monitor" },
      { name: "description", content: "Automatic number plate recognition on uploaded images." },
    ],
  }),
  component: ANPR,
});

function PlateCard({ raw, cleaned, conf, engine }: { raw: string; cleaned: string; conf: number; engine: string }) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    navigator.clipboard.writeText(cleaned);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-surface-border bg-surface-deep p-4"
    >
      <div className="flex items-center justify-between gap-3">
        <span className="font-mono text-2xl font-bold tracking-[0.2em] text-brand">
          {cleaned || "—"}
        </span>
        <Button size="icon" variant="ghost" onClick={copy}>
          {copied ? <Check className="h-4 w-4 text-status-live" /> : <Copy className="h-4 w-4" />}
        </Button>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
        <span className="font-mono">raw: "{raw}"</span>
        <Badge variant="secondary">{Math.round(conf * 100)}%</Badge>
        <Badge variant="secondary">{engine}</Badge>
      </div>
    </motion.div>
  );
}

function ReadTab() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ANPRResult | null>(null);

  const run = async () => {
    if (!file) return toast.error("Upload an image first");
    setLoading(true);
    setResult(null);
    try {
      setResult(await readPlates(file));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "ANPR failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <MotionCard title="Upload Image">
        <ImageUploader onFile={setFile} />
        <Button className="mt-4 w-full" onClick={run} disabled={loading}>
          {loading ? "Reading…" : "Read →"}
        </Button>
      </MotionCard>
      <MotionCard title={result ? `Plates Found: ${result.plate_count}` : "Plates"}>
        {result ? (
          <div className="space-y-3">
            {result.plates.map((p, i) => (
              <PlateCard
                key={i}
                raw={p.raw_text}
                cleaned={p.cleaned_text}
                conf={p.confidence}
                engine={result.ocr_engine}
              />
            ))}
            {result.plates.length === 0 && (
              <p className="py-8 text-center text-sm text-muted-foreground">
                No plates detected.
              </p>
            )}
          </div>
        ) : (
          <p className="py-12 text-center text-sm text-muted-foreground">No result yet.</p>
        )}
      </MotionCard>
    </div>
  );
}

function AnnotatedTab() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [blob, setBlob] = useState<Blob | null>(null);

  const run = async () => {
    if (!file) return toast.error("Upload an image first");
    setLoading(true);
    setBlob(null);
    try {
      setBlob(await readPlatesAnnotated(file));
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "ANPR failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <MotionCard title="Upload Image">
        <ImageUploader onFile={setFile} />
        <Button className="mt-4 w-full" onClick={run} disabled={loading}>
          {loading ? "Reading…" : "Read →"}
        </Button>
      </MotionCard>
      <MotionCard title="Annotated">
        <AnnotatedImage blob={blob} filename="anpr-annotated.jpg" />
      </MotionCard>
    </div>
  );
}

function ANPR() {
  return (
    <PageWrapper>
      <PageHeader title="ANPR" endpoint="POST /anpr/read · /anpr/read/annotated" />
      <Tabs defaultValue="read">
        <TabsList>
          <TabsTrigger value="read">Read Plates</TabsTrigger>
          <TabsTrigger value="annotated">Annotated</TabsTrigger>
        </TabsList>
        <TabsContent value="read">
          <ReadTab />
        </TabsContent>
        <TabsContent value="annotated">
          <AnnotatedTab />
        </TabsContent>
      </Tabs>
    </PageWrapper>
  );
}

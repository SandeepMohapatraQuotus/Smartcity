import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { motion } from "framer-motion";
import { toast } from "sonner";
import { PageWrapper, PageHeader } from "@/components/layout/PageHeader";
import { MotionCard } from "@/components/shared/MotionCard";
import { ImageUploader } from "@/components/shared/ImageUploader";
import { AnnotatedImage } from "@/components/shared/AnnotatedImage";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  detectVehicles,
  detectPersons,
  analyseFrameAnnotated,
} from "@/api/endpoints";
import type { VehicleDetectionResult, PersonDetectionResult } from "@/api/types";

export const Route = createFileRoute("/detect")({
  head: () => ({
    meta: [
      { title: "Detection — Smart City Monitor" },
      { name: "description", content: "Detect vehicles and persons in an uploaded image." },
    ],
  }),
  component: Detect,
});

const rowMotion = {
  hidden: { opacity: 0, x: -12 },
  show: (i: number) => ({ opacity: 1, x: 0, transition: { delay: i * 0.04 } }),
};

function VehiclesTab() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<VehicleDetectionResult | null>(null);
  const [annotated, setAnnotated] = useState<Blob | null>(null);

  const run = async () => {
    if (!file) return toast.error("Upload an image first");
    setLoading(true);
    setResult(null);
    setAnnotated(null);
    try {
      const [json, blob] = await Promise.all([
        detectVehicles(file),
        analyseFrameAnnotated(file),
      ]);
      setResult(json);
      setAnnotated(blob);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Detection failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <MotionCard title="Upload Image">
        <ImageUploader onFile={setFile} />
        <Button className="mt-4 w-full" onClick={run} disabled={loading}>
          {loading ? "Detecting…" : "Detect →"}
        </Button>
      </MotionCard>
      <MotionCard title="Output">
        <AnnotatedImage blob={annotated} filename="vehicles.jpg" />
        {result && (
          <div className="mt-4 space-y-3">
            <div className="flex flex-wrap gap-2">
              {Object.entries(result.vehicle_count).map(([k, v]) => (
                <Badge key={k} variant="secondary">
                  {k} {v}
                </Badge>
              ))}
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>track</TableHead>
                  <TableHead>label</TableHead>
                  <TableHead>conf</TableHead>
                  <TableHead>bbox</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {result.detections.map((d, i) => (
                  <motion.tr
                    key={d.track_id + "-" + i}
                    custom={i}
                    variants={rowMotion}
                    initial="hidden"
                    animate="show"
                    className="border-b border-surface-border"
                  >
                    <TableCell className="font-mono">{d.track_id}</TableCell>
                    <TableCell>{d.label}</TableCell>
                    <TableCell className="font-mono">
                      {Math.round(d.confidence * 100)}%
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      [{d.bbox.map((n) => Math.round(n)).join(", ")}]
                    </TableCell>
                  </motion.tr>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </MotionCard>
    </div>
  );
}

function PersonsTab() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PersonDetectionResult | null>(null);
  const [annotated, setAnnotated] = useState<Blob | null>(null);

  const run = async () => {
    if (!file) return toast.error("Upload an image first");
    setLoading(true);
    setResult(null);
    setAnnotated(null);
    try {
      const [json, blob] = await Promise.all([
        detectPersons(file),
        analyseFrameAnnotated(file),
      ]);
      setResult(json);
      setAnnotated(blob);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Detection failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <MotionCard title="Upload Image">
        <ImageUploader onFile={setFile} />
        <Button className="mt-4 w-full" onClick={run} disabled={loading}>
          {loading ? "Detecting…" : "Detect →"}
        </Button>
      </MotionCard>
      <MotionCard title="Output">
        <AnnotatedImage blob={annotated} filename="persons.jpg" />
        {result && (
          <div className="mt-4 space-y-3">
            <div className="text-center">
              <motion.div
                key={result.person_count}
                initial={{ scale: 0.6, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="font-mono text-5xl font-bold text-status-live"
              >
                {result.person_count}
              </motion.div>
              <div className="text-xs uppercase tracking-wider text-muted-foreground">
                persons detected
              </div>
            </div>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>track</TableHead>
                  <TableHead>conf</TableHead>
                  <TableHead>bbox</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {result.detections.map((d, i) => (
                  <motion.tr
                    key={d.track_id + "-" + i}
                    custom={i}
                    variants={rowMotion}
                    initial="hidden"
                    animate="show"
                    className="border-b border-surface-border"
                  >
                    <TableCell className="font-mono">{d.track_id}</TableCell>
                    <TableCell className="font-mono">
                      {Math.round(d.confidence * 100)}%
                    </TableCell>
                    <TableCell className="font-mono text-xs text-muted-foreground">
                      [{d.bbox.map((n) => Math.round(n)).join(", ")}]
                    </TableCell>
                  </motion.tr>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </MotionCard>
    </div>
  );
}

function Detect() {
  return (
    <PageWrapper>
      <PageHeader title="Detection" endpoint="POST /detect/vehicles · /detect/persons" />
      <Tabs defaultValue="vehicles">
        <TabsList>
          <TabsTrigger value="vehicles">Vehicles</TabsTrigger>
          <TabsTrigger value="persons">Persons</TabsTrigger>
        </TabsList>
        <TabsContent value="vehicles">
          <VehiclesTab />
        </TabsContent>
        <TabsContent value="persons">
          <PersonsTab />
        </TabsContent>
      </Tabs>
    </PageWrapper>
  );
}

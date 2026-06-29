import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { toast } from "sonner";
import { PageWrapper, PageHeader } from "@/components/layout/PageHeader";
import { MotionCard } from "@/components/shared/MotionCard";
import { ImageUploader } from "@/components/shared/ImageUploader";
import { JsonViewer } from "@/components/shared/JsonViewer";
import { AnnotatedImage } from "@/components/shared/AnnotatedImage";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { analyseFrame, analyseFrameAnnotated } from "@/api/endpoints";
import type { FrameEvent } from "@/api/types";

export const Route = createFileRoute("/analyse")({
  head: () => ({
    meta: [
      { title: "Full Pipeline Analysis — Smart City Monitor" },
      { name: "description", content: "Run the complete AI pipeline on a single uploaded image." },
    ],
  }),
  component: Analyse,
});

function Analyse() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<FrameEvent | null>(null);
  const [annotated, setAnnotated] = useState<Blob | null>(null);

  const run = async () => {
    if (!file) {
      toast.error("Upload an image first");
      return;
    }
    setLoading(true);
    setResult(null);
    setAnnotated(null);
    try {
      const [json, blob] = await Promise.all([
        analyseFrame(file),
        analyseFrameAnnotated(file),
      ]);
      setResult(json);
      setAnnotated(blob);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Analysis failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageWrapper>
      <PageHeader
        title="Full Pipeline Analysis"
        endpoint="POST /analyse/frame"
        description="Day/night, detection, ANPR and face recognition in one pass."
      />
      <div className="grid gap-6 lg:grid-cols-2">
        <MotionCard title="Input">
          <ImageUploader onFile={setFile} />
          <Button className="mt-4 w-full" onClick={run} disabled={loading}>
            {loading ? "Analysing…" : "Analyse →"}
          </Button>
        </MotionCard>

        <MotionCard title="Response">
          {loading ? (
            <div className="space-y-2">
              <Skeleton className="h-6 w-3/4" />
              <Skeleton className="h-6 w-1/2" />
              <Skeleton className="h-40 w-full" />
            </div>
          ) : result || annotated ? (
            <Tabs defaultValue="json">
              <TabsList>
                <TabsTrigger value="json">JSON</TabsTrigger>
                <TabsTrigger value="image">Annotated</TabsTrigger>
              </TabsList>
              <TabsContent value="json">
                {result && <JsonViewer data={result} />}
              </TabsContent>
              <TabsContent value="image">
                <AnnotatedImage blob={annotated} filename="analyse-annotated.jpg" />
              </TabsContent>
            </Tabs>
          ) : (
            <p className="py-12 text-center text-sm text-muted-foreground">
              Upload an image and run the pipeline to see results.
            </p>
          )}
        </MotionCard>
      </div>
    </PageWrapper>
  );
}

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Download, ImageOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { downloadBlob } from "@/lib/utils";

interface AnnotatedImageProps {
  blob: Blob | null;
  filename?: string;
  className?: string;
  downloadable?: boolean;
}

export function AnnotatedImage({
  blob,
  filename = "annotated.jpg",
  className,
  downloadable = true,
}: AnnotatedImageProps) {
  const [url, setUrl] = useState<string | null>(null);

  useEffect(() => {
    if (!blob) {
      setUrl(null);
      return;
    }
    const u = URL.createObjectURL(blob);
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [blob]);

  if (!url) {
    return (
      <div
        className={cn(
          "flex min-h-48 flex-col items-center justify-center rounded-lg border border-dashed border-surface-border bg-surface-deep text-muted-foreground",
          className,
        )}
      >
        <ImageOff className="mb-2 h-8 w-8" />
        <span className="text-sm">No annotated output yet</span>
      </div>
    );
  }

  return (
    <div className={cn("space-y-3", className)}>
      <motion.img
        key={url}
        src={url}
        alt="annotated result"
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
        className="w-full rounded-lg border border-surface-border object-contain"
      />
      {downloadable && blob && (
        <Button
          variant="secondary"
          size="sm"
          onClick={() => downloadBlob(blob, filename)}
        >
          <Download className="mr-2 h-4 w-4" /> Download
        </Button>
      )}
    </div>
  );
}

import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import { UploadCloud, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface ImageUploaderProps {
  onFile: (file: File | null) => void;
  accept?: string;
  label?: string;
  className?: string;
}

export function ImageUploader({
  onFile,
  accept = "image/*",
  label = "Drop image here or click to upload",
  className,
}: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handle = useCallback(
    (file: File | null) => {
      setPreview((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return file ? URL.createObjectURL(file) : null;
      });
      onFile(file);
    },
    [onFile],
  );

  return (
    <div className={cn("space-y-3", className)}>
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files?.[0];
          if (f) handle(f);
        }}
        className={cn(
          "relative flex min-h-48 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border-2 border-dashed border-surface-border bg-surface-deep p-6 text-center transition-colors",
          dragging && "border-brand bg-brand/5",
        )}
      >
        {preview ? (
          <>
            <motion.img
              key={preview}
              src={preview}
              alt="preview"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ type: "spring", stiffness: 260, damping: 18 }}
              className="max-h-72 w-full rounded-lg object-contain"
            />
            <Button
              size="icon"
              variant="secondary"
              className="absolute right-2 top-2 h-7 w-7"
              onClick={(e) => {
                e.stopPropagation();
                handle(null);
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          </>
        ) : (
          <>
            <UploadCloud className="mb-3 h-10 w-10 text-brand" />
            <p className="text-sm text-muted-foreground">{label}</p>
          </>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => handle(e.target.files?.[0] ?? null)}
      />
    </div>
  );
}

/**
 * ImageUploadModal
 * ─────────────────
 * Drag-and-drop / click-to-browse single-image upload modal.
 * Uploads to Cloudinary (dev, VITE_NODE_ENV !== "production") or S3 (prod).
 * On success calls onUploadSuccess(url) with the single uploaded image URL.
 *
 * Usage:
 *   <ImageUploadModal
 *     isOpen={open}
 *     onClose={() => setOpen(false)}
 *     onUploadSuccess={(url) => setImageUrl(url)}
 *   />
 */

import { useRef, useState, useEffect } from "react";
import { Upload, X, Loader2, ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { toast } from "sonner";

interface Props {
  isOpen: boolean;
  onClose: () => void;
  /** Called with the single uploaded URL on success. */
  onUploadSuccess: (url: string) => void;
}

// ── Cloudinary upload (dev) ───────────────────────────────────────────────────
async function uploadToCloudinary(file: File): Promise<string> {
  const cloudName = import.meta.env.VITE_CLOUDINARY_CLOUD_NAME;
  const uploadPreset = import.meta.env.VITE_CLOUDINARY_UPLOAD_PRESET;

  if (!cloudName || !uploadPreset) {
    throw new Error(
      "Cloudinary env vars not configured. " +
        "Set VITE_CLOUDINARY_CLOUD_NAME and VITE_CLOUDINARY_UPLOAD_PRESET in .env"
    );
  }

  const fd = new FormData();
  fd.append("file", file);
  fd.append("upload_preset", uploadPreset);
  fd.append("folder", "SmartCity-Watchlist");

  const res = await fetch(
    `https://api.cloudinary.com/v1_1/${cloudName}/image/upload`,
    { method: "POST", body: fd }
  );
  if (!res.ok) throw new Error(`Cloudinary upload failed: ${res.statusText}`);
  const data = await res.json();
  return data.secure_url as string;
}

// ── S3 pre-signed upload (prod) ───────────────────────────────────────────────
async function uploadToS3(file: File): Promise<string> {
  const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
  const res = await fetch(`${API_BASE}/upload/generate-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fileType: file.type }),
  });
  if (!res.ok) throw new Error("Failed to get S3 pre-signed URL");
  const { data } = await res.json();
  const { uploadUrl, fileUrl } = data;

  const put = await fetch(uploadUrl, {
    method: "PUT",
    headers: { "Content-Type": file.type },
    body: file,
  });
  if (!put.ok) throw new Error("S3 upload failed");
  return fileUrl as string;
}

async function uploadFile(file: File): Promise<string> {
  if (import.meta.env.VITE_NODE_ENV === "production") {
    return uploadToS3(file);
  }
  return uploadToCloudinary(file);
}

// ── Component ─────────────────────────────────────────────────────────────────
export default function ImageUploadModal({
  isOpen,
  onClose,
  onUploadSuccess,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Reset state when modal closes
  useEffect(() => {
    if (!isOpen) {
      if (preview) URL.revokeObjectURL(preview);
      setFile(null);
      setPreview(null);
      setError("");
      setIsDragging(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleFile = (f: File) => {
    if (!f.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setError("");
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }
    setIsUploading(true);
    setError("");
    try {
      const url = await uploadFile(file);
      toast.success("Image uploaded successfully");
      onUploadSuccess(url);
      onClose();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Upload failed";
      setError(msg);
      toast.error(msg);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Upload Profile Photo</DialogTitle>
          <DialogDescription>
            Upload a clear photo of the person. This image will be shown in
            alerts when this person is detected.
          </DialogDescription>
        </DialogHeader>

        {/* Drop zone */}
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={(e) => {
            e.preventDefault();
            setIsDragging(false);
          }}
          onDrop={handleDrop}
          className={cn(
            "relative flex min-h-52 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border-2 border-dashed transition-colors",
            isDragging
              ? "border-brand bg-brand/5"
              : "border-surface-border bg-surface-deep hover:border-brand/60 hover:bg-brand/5",
          )}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleInputChange}
          />

          {preview ? (
            <>
              <img
                src={preview}
                alt="Preview"
                className="h-full max-h-52 w-full rounded-xl object-cover"
              />
              {/* remove button */}
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  if (preview) URL.revokeObjectURL(preview);
                  setFile(null);
                  setPreview(null);
                  if (inputRef.current) inputRef.current.value = "";
                }}
                className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full bg-black/60 text-white transition-colors hover:bg-black/80"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </>
          ) : (
            <div className="flex flex-col items-center gap-3 text-center p-6">
              <div className="rounded-full bg-brand/10 p-4">
                {isDragging ? (
                  <Upload className="h-7 w-7 text-brand" />
                ) : (
                  <ImageIcon className="h-7 w-7 text-brand" />
                )}
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">
                  {isDragging ? "Drop to upload" : "Click or drag & drop"}
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  PNG, JPG, WEBP up to 10 MB
                </p>
              </div>
            </div>
          )}
        </div>

        {error && (
          <p className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {error}
          </p>
        )}

        <DialogFooter>
          <Button variant="ghost" onClick={onClose} disabled={isUploading}>
            Cancel
          </Button>
          <Button
            onClick={handleUpload}
            disabled={isUploading || !file}
            className="min-w-24"
          >
            {isUploading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Uploading…
              </>
            ) : (
              "Upload"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  CheckCircle2,
  CloudUpload,
  ImagePlus,
  Loader2,
  ShieldCheck,
  Trash2,
  UserPlus,
  UserX,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { PageWrapper, PageHeader } from "@/components/layout/PageHeader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
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
import {
  getWatchlist,
  addPersonToWatchlist,
  removeFromWatchlist,
} from "@/api/endpoints";
import type { AddPersonOutcome, WatchlistPerson } from "@/api/types";

export const Route = createFileRoute("/watchlist")({
  head: () => ({
    meta: [
      { title: "Watchlist — Smart City Monitor" },
      { name: "description", content: "Manage the face-recognition watchlist." },
    ],
  }),
  component: Watchlist,
});

// ─── Cloudinary uploader ─────────────────────────────────────────────────────
async function uploadToCloudinary(file: File): Promise<string> {
  const cloudName = import.meta.env.VITE_CLOUDINARY_CLOUD_NAME;
  const uploadPreset = import.meta.env.VITE_CLOUDINARY_UPLOAD_PRESET;

  if (!cloudName || !uploadPreset) {
    throw new Error(
      "Cloudinary is not configured — add VITE_CLOUDINARY_CLOUD_NAME and " +
        "VITE_CLOUDINARY_UPLOAD_PRESET to frontend/.env"
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

// ─── Step indicator ──────────────────────────────────────────────────────────
type Step = "idle" | "uploading_cloudinary" | "saving_backend" | "done";

function StepBadge({ step }: { step: Step }) {
  if (step === "idle") return null;

  const steps: { key: Step; label: string }[] = [
    { key: "uploading_cloudinary", label: "Uploading photo to Cloudinary…" },
    { key: "saving_backend", label: "Saving embeddings to database…" },
    { key: "done", label: "Done!" },
  ];

  return (
    <div className="space-y-2">
      {steps.map((s) => {
        const isActive = s.key === step;
        const isDone =
          steps.findIndex((x) => x.key === step) >
          steps.findIndex((x) => x.key === s.key);

        return (
          <div key={s.key} className="flex items-center gap-2 text-xs">
            {isDone ? (
              <CheckCircle2 className="h-4 w-4 shrink-0 text-green-500" />
            ) : isActive ? (
              <Loader2 className="h-4 w-4 shrink-0 animate-spin text-brand" />
            ) : (
              <div className="h-4 w-4 shrink-0 rounded-full border border-surface-border" />
            )}
            <span
              className={
                isDone
                  ? "text-green-500"
                  : isActive
                    ? "font-medium text-foreground"
                    : "text-muted-foreground"
              }
            >
              {s.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ─── Main component ──────────────────────────────────────────────────────────
function Watchlist() {
  const [people, setPeople] = useState<WatchlistPerson[]>([]);
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [images, setImages] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [nightAugment, setNightAugment] = useState(true);
  const [step, setStep] = useState<Step>("idle");
  const [outcome, setOutcome] = useState<AddPersonOutcome | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = async () => {
    try {
      const people = await getWatchlist();
      setPeople(people ?? []);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to load watchlist");
    }
  };

  useEffect(() => {
    load();
  }, []);

  const reset = () => {
    setName("");
    previews.forEach((p) => URL.revokeObjectURL(p));
    setImages([]);
    setPreviews([]);
    setNightAugment(true);
    setStep("idle");
    setOutcome(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  // When user picks files, generate local previews immediately
  const addFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    const added = Array.from(incoming).filter((f) =>
      f.type.startsWith("image/")
    );
    if (!added.length) return;
    const newPreviews = added.map((f) => URL.createObjectURL(f));
    setImages((prev) => [...prev, ...added]);
    setPreviews((prev) => [...prev, ...newPreviews]);
  };

  const removeImage = (i: number) => {
    URL.revokeObjectURL(previews[i]);
    setImages((prev) => prev.filter((_, idx) => idx !== i));
    setPreviews((prev) => prev.filter((_, idx) => idx !== i));
  };

  /**
   * THE UNIFIED SUBMIT FLOW:
   *  1. Upload first photo → Cloudinary → get image_url
   *  2. POST all photos + image_url → backend (embeddings + DB store)
   */
  const handleAdd = async () => {
    if (!name.trim()) return toast.error("Enter a name");
    if (images.length === 0) return toast.error("Add at least one photo");

    try {
      // ── Step 1: Upload first photo to Cloudinary ────────────────────────
      setStep("uploading_cloudinary");
      let imageUrl: string | null = null;
      try {
        imageUrl = await uploadToCloudinary(images[0]);
      } catch (cloudErr) {
        // Non-fatal: warn but continue without image URL
        console.warn("Cloudinary upload failed, continuing without image_url:", cloudErr);
        toast.warning("Photo upload to Cloudinary failed — person will be registered without a display photo.");
      }

      // ── Step 2: Send all files + image_url → backend ────────────────────
      setStep("saving_backend");
      const result = await addPersonToWatchlist({
        name: name.trim(),
        files: images,          // All files → face/body embeddings in backend
        nightAugment,
        imageUrl,               // Cloudinary URL → stored in persons.image_url
      });

      setStep("done");
      setOutcome(result);
      load();

      if (result.reused_existing_person) {
        toast.success(
          `Merged into existing "${result.name}" — ${result.face_embeddings_added}F / ${result.body_embeddings_added}B embeddings added`
        );
      } else if (result.face_embeddings_added > 0 || result.body_embeddings_added > 0) {
        toast.success(
          `Registered "${result.name}" — ${result.face_embeddings_added}F / ${result.body_embeddings_added}B embeddings${imageUrl ? " + photo saved" : ""}`
        );
      } else {
        toast.warning(
          `Partial registration: ${result.errors?.[0] ?? "no usable embeddings"}`
        );
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to register person");
      setStep("idle");
    }
  };

  const isSaving = step === "uploading_cloudinary" || step === "saving_backend";

  const remove = async (id: string) => {
    try {
      await removeFromWatchlist(id);
      setPeople((p) => p.filter((x) => x.person_id !== id));
      toast.success("Person removed");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to remove");
    }
  };

  return (
    <PageWrapper>
      <PageHeader
        title="Watchlist"
        endpoint="GET /watchlist · POST /watchlist/add · DELETE /watchlist/{id}"
        description={`${people.length} person${people.length === 1 ? "" : "s"} on watchlist`}
        action={
          <Dialog
            open={open}
            onOpenChange={(v) => {
              setOpen(v);
              if (!v) reset();
            }}
          >
            <DialogTrigger asChild>
              <Button>
                <UserPlus className="mr-2 h-4 w-4" /> Add Person
              </Button>
            </DialogTrigger>

            <DialogContent className="max-w-lg">
              <DialogHeader>
                <DialogTitle>Add to Watchlist</DialogTitle>
              </DialogHeader>

              {outcome ? (
                /* ── Success state ──────────────────────────────────────── */
                <div className="space-y-4 py-2">
                  <div className="flex items-center gap-3 rounded-lg border border-green-500/30 bg-green-500/10 px-4 py-3">
                    {outcome.image_url ? (
                      <img
                        src={outcome.image_url}
                        alt={outcome.name}
                        className="h-12 w-12 rounded-full object-cover ring-2 ring-green-500/40"
                      />
                    ) : (
                      <ShieldCheck className="h-8 w-8 text-green-400" />
                    )}
                    <div>
                      <p className="font-semibold">{outcome.name}</p>
                      <p className="font-mono text-xs text-muted-foreground">
                        {outcome.person_id}
                      </p>
                      {outcome.image_url && (
                        <p className="mt-0.5 flex items-center gap-1 text-xs text-green-500">
                          <CloudUpload className="h-3 w-3" />
                          Photo stored on Cloudinary
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-3 text-center">
                    {[
                      { label: "Images", value: outcome.images_received },
                      { label: "Face emb.", value: outcome.face_embeddings_added },
                      { label: "Body emb.", value: outcome.body_embeddings_added },
                    ].map((s) => (
                      <div
                        key={s.label}
                        className="rounded-lg border border-surface-border bg-muted/20 py-3"
                      >
                        <div className="font-mono text-2xl font-bold">{s.value}</div>
                        <div className="text-xs text-muted-foreground">{s.label}</div>
                      </div>
                    ))}
                  </div>

                  {outcome.images_skipped > 0 && (
                    <p className="text-center text-xs text-amber-400">
                      {outcome.images_skipped} image(s) had no detectable face or body
                    </p>
                  )}
                </div>
              ) : (
                /* ── Form state ─────────────────────────────────────────── */
                <div className="space-y-5">
                  {/* Name */}
                  <div className="space-y-1.5">
                    <Label>Name</Label>
                    <Input
                      id="watchlist-name"
                      placeholder="Full Name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      disabled={isSaving}
                    />
                  </div>

                  {/* Photo picker */}
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Photos</Label>
                      {images.length > 0 && (
                        <span className="text-xs text-muted-foreground">
                          {images.length} selected · first will be uploaded to Cloudinary as display photo
                        </span>
                      )}
                    </div>

                    {/* Previews grid */}
                    {previews.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        <AnimatePresence>
                          {previews.map((src, i) => (
                            <motion.div
                              key={src}
                              initial={{ opacity: 0, scale: 0.8 }}
                              animate={{ opacity: 1, scale: 1 }}
                              exit={{ opacity: 0, scale: 0.8 }}
                              className="relative"
                            >
                              <img
                                src={src}
                                alt={`photo-${i}`}
                                className={`h-16 w-16 rounded-lg object-cover ${
                                  i === 0
                                    ? "ring-2 ring-brand"   // first photo = the Cloudinary display one
                                    : "ring-1 ring-surface-border"
                                }`}
                              />
                              {/* "Display photo" badge on first image */}
                              {i === 0 && (
                                <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 whitespace-nowrap rounded-full bg-brand px-1.5 py-0.5 font-mono text-[9px] font-bold text-white">
                                  display
                                </span>
                              )}
                              {!isSaving && (
                                <button
                                  onClick={() => removeImage(i)}
                                  className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-status-alert text-white shadow"
                                >
                                  <X className="h-3 w-3" />
                                </button>
                              )}
                            </motion.div>
                          ))}
                        </AnimatePresence>
                      </div>
                    )}

                    {/* Drop zone / add more */}
                    {!isSaving && (
                      <button
                        id="watchlist-add-images-btn"
                        onClick={() => fileRef.current?.click()}
                        className="flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-surface-border py-4 text-sm text-muted-foreground transition-colors hover:border-brand hover:text-brand"
                      >
                        <ImagePlus className="h-4 w-4" />
                        {images.length === 0
                          ? "Click to add photos"
                          : "Add more photos"}
                      </button>
                    )}
                    <input
                      ref={fileRef}
                      type="file"
                      accept="image/*"
                      multiple
                      className="hidden"
                      onChange={(e) => addFiles(e.target.files)}
                    />

                    {images.length > 0 && !isSaving && (
                      <p className="text-xs text-muted-foreground">
                        <span className="font-semibold text-brand">First photo</span>{" "}
                        (highlighted) → uploaded to Cloudinary as display photo.
                        All photos → sent to backend for face/body embeddings.
                      </p>
                    )}
                  </div>

                  {/* Night augment */}
                  <div className="flex items-center gap-3">
                    <input
                      id="watchlist-night-augment"
                      type="checkbox"
                      checked={nightAugment}
                      onChange={(e) => setNightAugment(e.target.checked)}
                      disabled={isSaving}
                      className="h-4 w-4 rounded border-surface-border accent-brand"
                    />
                    <Label htmlFor="watchlist-night-augment" className="cursor-pointer">
                      Night augmentation
                      <span className="ml-1 text-xs text-muted-foreground">
                        (improves low-light matching)
                      </span>
                    </Label>
                  </div>

                  {/* Progress steps shown during save */}
                  {isSaving && <StepBadge step={step} />}
                </div>
              )}

              <DialogFooter>
                <Button
                  variant="ghost"
                  onClick={() => {
                    setOpen(false);
                    reset();
                  }}
                  disabled={isSaving}
                >
                  {outcome ? "Close" : "Cancel"}
                </Button>

                {!outcome && (
                  <Button
                    id="watchlist-submit-btn"
                    onClick={handleAdd}
                    disabled={isSaving || images.length === 0 || !name.trim()}
                  >
                    {isSaving ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        {step === "uploading_cloudinary"
                          ? "Uploading photo…"
                          : "Saving to DB…"}
                      </>
                    ) : (
                      "Add →"
                    )}
                  </Button>
                )}

                {outcome && (
                  <Button onClick={reset} variant="outline">
                    Add Another
                  </Button>
                )}
              </DialogFooter>
            </DialogContent>
          </Dialog>
        }
      />

      {/* ── Person grid ──────────────────────────────────────────────────── */}
      {people.length === 0 ? (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-surface-border py-20 text-center">
          <UserX className="mb-3 h-12 w-12 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">No persons on watchlist</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
          <AnimatePresence>
            {people.map((p) => (
              <motion.div
                key={p.person_id}
                layout
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex flex-col items-center rounded-xl border border-surface-border bg-card p-5 text-center"
              >
                {/* Avatar — real photo if Cloudinary URL exists */}
                {p.image_url ? (
                  <img
                    src={p.image_url}
                    alt={p.name}
                    className="h-16 w-16 rounded-full object-cover ring-2 ring-brand/30"
                  />
                ) : (
                  <div className="grid h-16 w-16 place-items-center rounded-full bg-brand/15 text-2xl">
                    👤
                  </div>
                )}

                <div className="mt-3 font-semibold">{p.name}</div>
                <div className="font-mono text-xs text-muted-foreground">
                  {p.person_id.slice(0, 8)}…
                </div>

                {/* Embedding counts */}
                {(p.face_refs !== undefined || p.body_refs !== undefined) && (
                  <div className="mt-1 flex gap-2 font-mono text-[10px] text-muted-foreground/70">
                    {p.face_refs !== undefined && (
                      <span>{p.face_refs}F</span>
                    )}
                    {p.body_refs !== undefined && (
                      <span>{p.body_refs}B</span>
                    )}
                  </div>
                )}

                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="mt-3 text-status-alert"
                    >
                      <Trash2 className="mr-1 h-4 w-4" /> Remove
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>Remove {p.name}?</AlertDialogTitle>
                      <AlertDialogDescription>
                        This removes all embeddings for {p.person_id} from the
                        face watchlist.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                      <AlertDialogAction onClick={() => remove(p.person_id)}>
                        Remove
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </PageWrapper>
  );
}

import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ImagePlus, ShieldCheck, Trash2, UserPlus, UserX, X } from "lucide-react";
import { toast } from "sonner";
import { PageWrapper, PageHeader } from "@/components/layout/PageHeader";
import { ImageUploader } from "@/components/shared/ImageUploader";
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

function Watchlist() {
  const [people, setPeople] = useState<WatchlistPerson[]>([]);
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [images, setImages] = useState<File[]>([]);
  const [nightAugment, setNightAugment] = useState(true);
  const [saving, setSaving] = useState(false);
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
    setImages([]);
    setNightAugment(true);
    setOutcome(null);
  };

  const addImages = (files: FileList | null) => {
    if (!files) return;
    setImages((prev) => [...prev, ...Array.from(files)]);
  };

  const removeImage = (i: number) =>
    setImages((prev) => prev.filter((_, idx) => idx !== i));

  const add = async () => {
    if (!name.trim()) return toast.error("Enter a name");
    if (images.length === 0) return toast.error("Add at least one photo");
    setSaving(true);
    try {
      const result = await addPersonToWatchlist({
        name: name.trim(),
        files: images,
        nightAugment,
      });
      setOutcome(result);
      load();
      if (result.reused_existing_person) {
        toast.success(
          `Merged into existing "${result.name}" (${result.face_embeddings_added}F / ${result.body_embeddings_added}B embeddings added)`,
        );
      } else if (result.face_embeddings_added > 0 || result.body_embeddings_added > 0) {
        toast.success(
          `Added "${result.name}" (${result.face_embeddings_added}F / ${result.body_embeddings_added}B embeddings)`,
        );
      } else {
        toast.warning(`Partial registration: ${result.errors?.[0] ?? "no usable embeddings"}`);
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to add");
    } finally {
      setSaving(false);
    }
  };

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
                /* ── Success state ─────────────────────────────── */
                <div className="space-y-4 py-2">
                  <div className="flex items-center gap-3 rounded-lg border border-green-500/30 bg-green-500/10 px-4 py-3">
                    <ShieldCheck className="h-6 w-6 text-green-400" />
                    <div>
                      <p className="font-semibold">{outcome.name}</p>
                      <p className="font-mono text-xs text-muted-foreground">
                        {outcome.person_id}
                      </p>
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
                /* ── Form state ────────────────────────────────── */
                <div className="space-y-4">
                  <div className="space-y-1.5">
                    <Label>Name</Label>
                    <Input
                      id="watchlist-name"
                      placeholder="Full Name"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                    />
                  </div>

                  <div className="space-y-1.5">
                    <Label>Photos</Label>
                    <p className="text-xs text-muted-foreground">
                      Add one or more face photos. More images improve matching accuracy.
                    </p>

                    {images.length > 0 && (
                      <div className="flex flex-wrap gap-2">
                        <AnimatePresence>
                          {images.map((img, i) => (
                            <motion.div
                              key={img.name + i}
                              initial={{ opacity: 0, scale: 0.8 }}
                              animate={{ opacity: 1, scale: 1 }}
                              exit={{ opacity: 0, scale: 0.8 }}
                              className="relative"
                            >
                              <img
                                src={URL.createObjectURL(img)}
                                alt={img.name}
                                className="h-16 w-16 rounded-md object-cover ring-1 ring-surface-border"
                              />
                              <button
                                onClick={() => removeImage(i)}
                                className="absolute -right-1.5 -top-1.5 flex h-5 w-5 items-center justify-center rounded-full bg-status-alert text-white shadow"
                              >
                                <X className="h-3 w-3" />
                              </button>
                            </motion.div>
                          ))}
                        </AnimatePresence>
                      </div>
                    )}

                    <button
                      id="watchlist-add-images-btn"
                      onClick={() => fileRef.current?.click()}
                      className="flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-surface-border py-4 text-sm text-muted-foreground transition-colors hover:border-brand hover:text-brand"
                    >
                      <ImagePlus className="h-4 w-4" />
                      {images.length === 0 ? "Add photos" : "Add more photos"}
                    </button>
                    <input
                      ref={fileRef}
                      type="file"
                      accept="image/*"
                      multiple
                      className="hidden"
                      onChange={(e) => addImages(e.target.files)}
                    />
                  </div>

                  {/* Night augment toggle */}
                  <div className="flex items-center gap-3">
                    <input
                      id="watchlist-night-augment"
                      type="checkbox"
                      checked={nightAugment}
                      onChange={(e) => setNightAugment(e.target.checked)}
                      className="h-4 w-4 rounded border-surface-border accent-brand"
                    />
                    <Label htmlFor="watchlist-night-augment" className="cursor-pointer">
                      Night augmentation
                      <span className="ml-1 text-xs text-muted-foreground">
                        (improves low-light matching)
                      </span>
                    </Label>
                  </div>
                </div>
              )}

              <DialogFooter>
                <Button
                  variant="ghost"
                  onClick={() => { setOpen(false); reset(); }}
                >
                  {outcome ? "Close" : "Cancel"}
                </Button>
                {!outcome && (
                  <Button
                    id="watchlist-submit-btn"
                    onClick={add}
                    disabled={saving}
                  >
                    {saving ? "Adding…" : "Add →"}
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
                <div className="grid h-16 w-16 place-items-center rounded-full bg-brand/15 text-2xl">
                  👤
                </div>
                <div className="mt-3 font-semibold">{p.name}</div>
                <div className="font-mono text-xs text-muted-foreground">
                  {p.person_id}
                </div>
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button variant="ghost" size="sm" className="mt-3 text-status-alert">
                      <Trash2 className="mr-1 h-4 w-4" /> Remove
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>Remove {p.name}?</AlertDialogTitle>
                      <AlertDialogDescription>
                        This removes {p.person_id} from the face watchlist.
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

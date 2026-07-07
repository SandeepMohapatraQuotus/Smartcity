import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  UserPlus,
  Trash2,
  UserX,
  ShieldCheck,
  ScanFace,
  ImagePlus,
  X,
  Fingerprint,
} from "lucide-react";
import { toast } from "sonner";
import { PageWrapper, PageHeader } from "@/components/layout/PageHeader";
import { MotionCard } from "@/components/shared/MotionCard";
import { ImageUploader } from "@/components/shared/ImageUploader";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  analyseIdentify,
  analyseIdentifyAnnotated,
  addRegistryPerson,
  removeRegistryPerson,
  listRegistryPeople,
} from "@/api/endpoints";
import type {
  PersonDetectionResult,
  IdentifiedPerson,
  IdentifyPersonResult,
  IdentifyResult,
  RegistryPerson,
  AddPersonOutcome,
} from "@/api/types";
import { AnnotatedImage } from "@/components/shared/AnnotatedImage";

export const Route = createFileRoute("/persons")({
  head: () => ({
    meta: [
      { title: "Persons — Smart City Monitor" },
      {
        name: "description",
        content:
          "Detect persons in frames and manage the person identity registry.",
      },
    ],
  }),
  component: Persons,
});

// ── Row animation ─────────────────────────────────────────────────
const rowMotion = {
  hidden: { opacity: 0, x: -12 },
  show: (i: number) => ({ opacity: 1, x: 0, transition: { delay: i * 0.05 } }),
};

// ─────────────────────────────────────────────────────────────────
// Tab 1: Detect Persons + Identity
// ─────────────────────────────────────────────────────────────────
function DetectTab() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<IdentifyResult | null>(null);
  const [annotated, setAnnotated] = useState<Blob | null>(null);

  const run = async () => {
    if (!file) return toast.error("Upload an image first");
    setLoading(true);
    setResult(null);
    setAnnotated(null);
    try {
      const [identifyResult, blob] = await Promise.all([
        analyseIdentify(file),
        analyseIdentifyAnnotated(file),
      ]);
      setResult(identifyResult);
      setAnnotated(blob);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Detection failed");
    } finally {
      setLoading(false);
    }
  };

  // people[] = body-detected persons (may or may not be identified)
  // unbound_faces[] = faces found whose centre wasn't inside any body bbox
  const identifiedWithBody = result?.people.filter((p) => p.name !== null) ?? [];
  const identifiedFaceOnly = result?.unbound_faces.filter((f) => f.name !== null) ?? [];
  const totalIdentified = identifiedWithBody.length + identifiedFaceOnly.length;

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <MotionCard title="Upload Image">
        <ImageUploader onFile={setFile} />
        <Button className="mt-4 w-full" onClick={run} disabled={loading}>
          {loading ? "Detecting…" : "Detect Persons →"}
        </Button>
      </MotionCard>

      <MotionCard title="Results">
        <AnnotatedImage blob={annotated} filename="persons.jpg" />

        {result && (
          <div className="mt-4 space-y-3">
            {/* ── Summary stats ───────────────────────────────── */}
            <div className="flex flex-wrap gap-3">
              <div className="flex flex-1 items-center justify-between rounded-lg border border-surface-border bg-muted/30 px-4 py-3">
                <span className="text-sm text-muted-foreground">Bodies</span>
                <motion.span
                  key={result.person_count}
                  initial={{ scale: 0.6, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  className="font-mono text-3xl font-bold text-status-live"
                >
                  {result.person_count}
                </motion.span>
              </div>
              {result.unbound_faces.length > 0 && (
                <div className="flex flex-1 items-center justify-between rounded-lg border border-amber-500/30 bg-amber-500/10 px-4 py-3">
                  <span className="text-sm text-amber-400">Face only</span>
                  <motion.span
                    key={result.unbound_faces.length}
                    initial={{ scale: 0.6, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="font-mono text-3xl font-bold text-amber-400"
                  >
                    {result.unbound_faces.length}
                  </motion.span>
                </div>
              )}
              {totalIdentified > 0 && (
                <div className="flex flex-1 items-center justify-between rounded-lg border border-green-500/30 bg-green-500/10 px-4 py-3">
                  <span className="text-sm text-green-400">Identified</span>
                  <motion.span
                    key={totalIdentified}
                    initial={{ scale: 0.6, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="font-mono text-3xl font-bold text-green-400"
                  >
                    {totalIdentified}
                  </motion.span>
                </div>
              )}
            </div>

            {/* ── Unified results table ───────────────────────── */}
            {(result.people.length > 0 || result.unbound_faces.length > 0) && (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>track / source</TableHead>
                    <TableHead>identity</TableHead>
                    <TableHead>sim</TableHead>
                    <TableHead>via</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {/* Body-detected persons */}
                  {result.people.map((p, i) => (
                    <motion.tr
                      key={`body-${p.track_id}-${i}`}
                      custom={i}
                      variants={rowMotion}
                      initial="hidden"
                      animate="show"
                      className="border-b border-surface-border"
                    >
                      <TableCell className="font-mono text-xs">
                        {p.track_id !== null ? (
                          <span className="rounded bg-muted/40 px-1.5 py-0.5">
                            #{p.track_id}
                          </span>
                        ) : "—"}
                      </TableCell>
                      <TableCell>
                        {p.name ? (
                          <span className="font-semibold text-green-400">{p.name}</span>
                        ) : (
                          <span className="text-muted-foreground">Unknown</span>
                        )}
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {p.similarity !== null ? (
                          <span className={p.method === "face" ? "text-green-400" : "text-cyan-400"}>
                            {((p.similarity ?? 0) * 100).toFixed(1)}%
                          </span>
                        ) : "—"}
                      </TableCell>
                      <TableCell>
                        {p.method ? (
                          <Badge className={`text-[10px] border-0 ${
                            p.method === "face"
                              ? "bg-green-500/20 text-green-400"
                              : "bg-cyan-500/20 text-cyan-400"
                          }`}>
                            {p.method}
                          </Badge>
                        ) : "—"}
                      </TableCell>
                    </motion.tr>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>
        )}
      </MotionCard>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// Add Person Dialog
// ─────────────────────────────────────────────────────────────────
function AddPersonDialog({ onAdded }: { onAdded: () => void }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [images, setImages] = useState<File[]>([]);
  const [saving, setSaving] = useState(false);
  const [outcome, setOutcome] = useState<AddPersonOutcome | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const reset = () => {
    setName("");
    setImages([]);
    setOutcome(null);
  };

  const addImages = (files: FileList | null) => {
    if (!files) return;
    setImages((prev) => [...prev, ...Array.from(files)]);
  };

  const removeImage = (i: number) =>
    setImages((prev) => prev.filter((_, idx) => idx !== i));

  const submit = async () => {
    if (!name.trim()) return toast.error("Enter a name");
    if (images.length === 0) return toast.error("Add at least one image");
    setSaving(true);
    try {
      const result = await addRegistryPerson(name.trim(), images);
      setOutcome(result);
      onAdded();
      if (result.reused_existing_person) {
        toast.success(
          `Merged into existing "${result.name}" (${result.face_embeddings_added}F / ${result.body_embeddings_added}B embeddings added)`
        );
      } else if (result.face_embeddings_added > 0 || result.body_embeddings_added > 0) {
        toast.success(
          `Registered "${result.name}" (${result.face_embeddings_added}F / ${result.body_embeddings_added}B embeddings)`
        );
      } else {
        toast.warning(
          `Partial registration: ${result.errors?.[0] ?? "no usable embeddings"}`
        );
      }
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Registration failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        setOpen(v);
        if (!v) reset();
      }}
    >
      <DialogTrigger asChild>
        <Button>
          <UserPlus className="mr-2 h-4 w-4" /> Register Person
        </Button>
      </DialogTrigger>

      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Register Person</DialogTitle>
        </DialogHeader>

        {outcome ? (
          /* Success state */
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
          /* Form state */
          <div className="space-y-4">
            <div className="space-y-1.5">
              <Label>Name</Label>
              <Input
                id="reg-name"
                placeholder="Full Name"
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </div>

            <div className="space-y-1.5">
              <Label>Reference Images</Label>
              <p className="text-xs text-muted-foreground">
                Add face photos, full-body shots, or a mix. More images = better
                matching.
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
                id="add-images-btn"
                onClick={() => fileRef.current?.click()}
                className="flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-surface-border py-4 text-sm text-muted-foreground transition-colors hover:border-brand hover:text-brand"
              >
                <ImagePlus className="h-4 w-4" />
                {images.length === 0 ? "Add images" : "Add more images"}
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
          </div>
        )}

        <DialogFooter>
          <Button variant="ghost" onClick={() => { setOpen(false); reset(); }}>
            {outcome ? "Close" : "Cancel"}
          </Button>
          {!outcome && (
            <Button onClick={submit} disabled={saving} id="submit-register-btn">
              {saving ? "Registering…" : "Register →"}
            </Button>
          )}
          {outcome && (
            <Button onClick={reset} variant="outline">
              Register Another
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ─────────────────────────────────────────────────────────────────
// Tab 2: Person Registry (list + add + delete)
// ─────────────────────────────────────────────────────────────────
function RegistryTab() {
  const [people, setPeople] = useState<RegistryPerson[]>([]);
  const [loading, setLoading] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const list = await listRegistryPeople();
      setPeople(list ?? []);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to load registry");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const remove = async (id: string) => {
    try {
      await removeRegistryPerson(id);
      setPeople((p) => p.filter((x) => x.person_id !== id));
      toast.success("Person removed from registry");
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to remove");
    }
  };

  const totalFace = people.reduce((s, p) => s + p.face_refs, 0);
  const totalBody = people.reduce((s, p) => s + p.body_refs, 0);

  return (
    <div className="space-y-6">
      {/* Header row */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap gap-3">
          <div className="flex items-center gap-2 rounded-lg border border-surface-border bg-muted/20 px-3 py-1.5 text-sm">
            <ScanFace className="h-4 w-4 text-brand" />
            <span className="font-mono font-semibold">{totalFace}</span>
            <span className="text-muted-foreground">face embeddings</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-surface-border bg-muted/20 px-3 py-1.5 text-sm">
            <Fingerprint className="h-4 w-4 text-status-live" />
            <span className="font-mono font-semibold">{totalBody}</span>
            <span className="text-muted-foreground">body embeddings</span>
          </div>
        </div>
        <AddPersonDialog onAdded={load} />
      </div>

      {/* Empty state */}
      {!loading && people.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-surface-border py-20 text-center">
          <UserX className="mb-3 h-12 w-12 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">No persons registered yet</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Use "Register Person" to add someone to the identity database
          </p>
        </div>
      )}

      {/* Person cards grid */}
      {people.length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <AnimatePresence>
            {people.map((p, i) => (
              <motion.div
                key={p.person_id}
                layout
                custom={i}
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0, transition: { delay: i * 0.04 } }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="flex flex-col rounded-xl border border-surface-border bg-card p-5"
              >
                {/* Avatar + name */}
                <div className="flex items-center gap-3">
                  <div className="grid h-12 w-12 shrink-0 place-items-center rounded-full bg-brand/15 text-xl">
                    👤
                  </div>
                  <div className="min-w-0">
                    <p className="truncate font-semibold">{p.name}</p>
                    <p className="truncate font-mono text-xs text-muted-foreground">
                      {p.person_id}
                    </p>
                  </div>
                </div>

                {/* Embedding counts */}
                <div className="mt-4 flex gap-2">
                  <Badge variant="secondary" className="gap-1">
                    <ScanFace className="h-3 w-3" />
                    {p.face_refs} face
                  </Badge>
                  <Badge variant="outline" className="gap-1">
                    <Fingerprint className="h-3 w-3" />
                    {p.body_refs} body
                  </Badge>
                </div>

                {/* Delete */}
                <div className="mt-4 flex justify-end">
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-status-alert hover:bg-status-alert/10"
                        id={`delete-${p.person_id}`}
                      >
                        <Trash2 className="mr-1 h-4 w-4" /> Remove
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Remove {p.name}?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will permanently delete {p.name} and all their face
                          and body embeddings from the registry. This cannot be undone.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={() => remove(p.person_id)}
                          className="bg-status-alert text-white hover:bg-status-alert/90"
                        >
                          Remove
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────
// Root Page
// ─────────────────────────────────────────────────────────────────
function Persons() {
  return (
    <PageWrapper>
      <PageHeader
        title="Person Detection & Registry"
        endpoint="POST /analyse/frame · POST /person/add · DELETE /person/{id} · GET /person"
        description="Detect and identify people in frames, manage the pgvector identity registry"
      />

      <Tabs defaultValue="detect">
        <TabsList>
          <TabsTrigger value="detect" id="tab-detect">
            Detect
          </TabsTrigger>
          <TabsTrigger value="registry" id="tab-registry">
            Identity Registry
          </TabsTrigger>
        </TabsList>

        <TabsContent value="detect" className="mt-6">
          <DetectTab />
        </TabsContent>

        <TabsContent value="registry" className="mt-6">
          <RegistryTab />
        </TabsContent>
      </Tabs>
    </PageWrapper>
  );
}

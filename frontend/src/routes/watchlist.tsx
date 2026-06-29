import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { UserPlus, Trash2, UserX } from "lucide-react";
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
  addToWatchlist,
  removeFromWatchlist,
} from "@/api/endpoints";
import type { WatchlistPerson } from "@/api/types";

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
  const [personId, setPersonId] = useState("");
  const [name, setName] = useState("");
  const [photo, setPhoto] = useState<File | null>(null);
  const [saving, setSaving] = useState(false);

  const load = async () => {
    try {
      setPeople(await getWatchlist());
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "Failed to load watchlist");
    }
  };

  useEffect(() => {
    load();
  }, []);

  const add = async () => {
    if (!personId.trim() || !name.trim() || !photo) {
      toast.error("Fill all fields and add a photo");
      return;
    }
    setSaving(true);
    try {
      await addToWatchlist(personId.trim(), name.trim(), photo);
      toast.success("Person added");
      setOpen(false);
      setPersonId("");
      setName("");
      setPhoto(null);
      load();
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
          <Dialog open={open} onOpenChange={setOpen}>
            <DialogTrigger asChild>
              <Button>
                <UserPlus className="mr-2 h-4 w-4" /> Add Person
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Add to Watchlist</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div className="space-y-1.5">
                  <Label>Person ID</Label>
                  <Input
                    placeholder="P004"
                    value={personId}
                    onChange={(e) => setPersonId(e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Name</Label>
                  <Input
                    placeholder="Full Name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label>Photo</Label>
                  <ImageUploader onFile={setPhoto} label="Drop a face photo" />
                </div>
              </div>
              <DialogFooter>
                <Button variant="ghost" onClick={() => setOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={add} disabled={saving}>
                  {saving ? "Adding…" : "Add →"}
                </Button>
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

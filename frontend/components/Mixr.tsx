"use client";

import { useRef, useState } from "react";

import {
  ApiError,
  analyzeTracks,
  createMix,
  orderTracks,
  uploadTracks,
  type MixResult as Mix,
  type Track,
} from "@/lib/api";

import Dropzone from "./Dropzone";
import MixResult from "./MixResult";
import Stepper from "./Stepper";
import TrackCard from "./TrackCard";
import Button from "./ui/Button";
import LoadingPanel from "./ui/LoadingPanel";

const MIN_TRACKS = 3;
const MAX_TRACKS = 5;
const ALLOWED_EXTENSIONS = [".mp3", ".wav"];
const MIN_LOADING_MS = 700;

type Stage = "select" | "uploaded" | "analyzed" | "ordered" | "mixed";
type Job = "upload" | "analyze" | "order" | "mix";

const STAGES: Stage[] = ["select", "uploaded", "analyzed", "ordered", "mixed"];
const STEP_LABELS = ["Upload", "Analyze", "Order", "Mix", "Listen"];

const LOADING: Record<Job, { title: string; detail: string }> = {
  upload: { title: "Uploading your tracks", detail: "Making sure every file is real audio" },
  analyze: { title: "Listening closely", detail: "Reading tempo, beats, key and energy" },
  order: { title: "Finding the best order", detail: "Scoring how each song flows into the next" },
  mix: { title: "Mixing", detail: "Blending every transition" },
};

const HEADINGS: Record<Exclude<Stage, "mixed">, { title: string; detail: string }> = {
  select: {
    title: "Start a new mix",
    detail: `Pick ${MIN_TRACKS} to ${MAX_TRACKS} songs you want blended together.`,
  },
  uploaded: { title: "Your tracks", detail: "Next, Mixr analyzes each song." },
  analyzed: { title: "Analyzed", detail: "Next, Mixr finds the order that flows best." },
  ordered: { title: "Your running order", detail: "Ready to mix." },
};

function validateFiles(files: File[]): string | null {
  const bad = files.find(
    (f) => !ALLOWED_EXTENSIONS.some((ext) => f.name.toLowerCase().endsWith(ext)),
  );
  if (bad) return `${bad.name}: only MP3 and WAV are supported`;
  if (files.length < MIN_TRACKS || files.length > MAX_TRACKS) {
    return `Pick between ${MIN_TRACKS} and ${MAX_TRACKS} songs (you picked ${files.length})`;
  }
  return null;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export default function Mixr() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [mixResult, setMixResult] = useState<Mix | null>(null);
  const [stage, setStage] = useState<Stage>("select");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const trackIds = tracks.map((t) => t.id);
  const filesError = files.length > 0 ? validateFiles(files) : null;

  function pickFiles(list: FileList | null) {
    if (!list) return;
    setFiles(Array.from(list));
    setError(null);
  }

  async function run(next: Job, action: () => Promise<void>) {
    setJob(next);
    setError(null);
    setNotice(null);
    try {
      await Promise.all([action(), sleep(MIN_LOADING_MS)]);
    } catch (e) {
      if (e instanceof ApiError && e.status === 501) setNotice(e.message);
      else setError(e instanceof Error ? e.message : String(e));
    } finally {
      setJob(null);
    }
  }

  const upload = () =>
    run("upload", async () => {
      const res = await uploadTracks(files);
      setTracks(res.tracks);
      setStage("uploaded");
    });

  const analyze = () =>
    run("analyze", async () => {
      const res = await analyzeTracks(trackIds);
      setTracks(res.tracks);
      setStage("analyzed");
    });

  const order = () =>
    run("order", async () => {
      const res = await orderTracks(trackIds);
      const byId = new Map(tracks.map((t) => [t.id, t]));
      setTracks(res.ordered_track_ids.map((id) => byId.get(id)!));
      setStage("ordered");
    });

  const mix = () =>
    run("mix", async () => {
      setMixResult(await createMix(trackIds));
      setStage("mixed");
    });

  function reset() {
    setFiles([]);
    setTracks([]);
    setMixResult(null);
    setStage("select");
    setError(null);
    setNotice(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  const nextAction = {
    uploaded: { label: "Analyze tracks", onClick: analyze },
    analyzed: { label: "Find the best order", onClick: order },
    ordered: { label: "Mix it", onClick: mix },
  }[stage as "uploaded" | "analyzed" | "ordered"];

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-8 px-6 pt-4 pb-16">
      <Stepper steps={STEP_LABELS} current={STAGES.indexOf(stage)} />

      {stage !== "mixed" && !job && (
        <div className="animate-fade-up flex flex-col gap-1">
          <h1 className="font-display text-3xl font-semibold">{HEADINGS[stage].title}</h1>
          <p className="text-neutral-400">{HEADINGS[stage].detail}</p>
        </div>
      )}

      {job ? (
        <LoadingPanel {...LOADING[job]} />
      ) : stage === "select" ? (
        <section className="flex flex-col gap-4">
          <Dropzone ref={inputRef} min={MIN_TRACKS} max={MAX_TRACKS} onFiles={pickFiles} />

          {files.length > 0 && (
            <ul className="glass flex flex-col divide-y divide-white/5 rounded-2xl px-4 text-sm">
              {files.map((f) => (
                <li key={f.name + f.size} className="flex justify-between gap-4 py-2.5">
                  <span className="truncate text-neutral-200">{f.name}</span>
                  <span className="shrink-0 font-mono text-neutral-500">
                    {(f.size / 1024 / 1024).toFixed(1)} MB
                  </span>
                </li>
              ))}
            </ul>
          )}

          {filesError && <p className="text-sm text-amber-300">{filesError}</p>}

          <Button onClick={upload} disabled={files.length === 0 || !!filesError}>
            Upload {files.length > 0 && !filesError ? `${files.length} songs` : ""}
          </Button>
        </section>
      ) : (
        <section className="flex flex-col gap-4">
          {stage === "mixed" && mixResult ? (
            <MixResult mix={mixResult} tracks={tracks} />
          ) : (
            <ul className="flex flex-col gap-3">
              {tracks.map((track, i) => (
                <TrackCard key={track.id} track={track} position={i + 1} />
              ))}
            </ul>
          )}

          {stage === "analyzed" && tracks.every((t) => t.analysis_status === "pending") && (
            <p className="text-sm text-neutral-500">
              Audio analysis isn&apos;t connected yet, so BPM and key aren&apos;t shown.
            </p>
          )}

          <div className="flex flex-col gap-3 sm:flex-row">
            {nextAction && (
              <Button onClick={nextAction.onClick} className="flex-1">
                {nextAction.label}
              </Button>
            )}
            <Button variant="ghost" onClick={reset} className={nextAction ? "" : "flex-1"}>
              {stage === "mixed" ? "Make another mix" : "Start over"}
            </Button>
          </div>
        </section>
      )}

      {notice && !job && (
        <p className="glass rounded-2xl px-4 py-3 text-sm text-neutral-300">
          <span className="text-neon font-semibold">Coming soon:</span> {notice}
        </p>
      )}
      {error && !job && (
        <p className="rounded-2xl border border-red-400/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </p>
      )}
    </div>
  );
}

"use client";

import { useRef, useState } from "react";

import {
  ApiError,
  analyzeTracks,
  createMix,
  orderTracks,
  trackAudioUrl,
  uploadTracks,
  type MixResult as Mix,
  type Track,
} from "@/lib/api";
import { formatSeconds } from "@/lib/format";

import MixResult from "./MixResult";

const MIN_TRACKS = 3;
const MAX_TRACKS = 5;
const ALLOWED_EXTENSIONS = [".mp3", ".wav"];

type Stage = "select" | "uploaded" | "analyzed" | "ordered" | "mixed";

const STEPS: { stage: Stage; label: string }[] = [
  { stage: "select", label: "Upload" },
  { stage: "uploaded", label: "Analyze" },
  { stage: "analyzed", label: "Order" },
  { stage: "ordered", label: "Mix" },
  { stage: "mixed", label: "Listen" },
];

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

export default function Mixr() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [mixResult, setMixResult] = useState<Mix | null>(null);
  const [stage, setStage] = useState<Stage>("select");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);

  const trackIds = tracks.map((t) => t.id);
  const filesError = files.length > 0 ? validateFiles(files) : null;

  function pickFiles(list: FileList | null) {
    if (!list) return;
    setFiles(Array.from(list));
    setError(null);
  }

  async function run(label: string, action: () => Promise<void>) {
    setBusy(label);
    setError(null);
    setNotice(null);
    try {
      await action();
    } catch (e) {
      if (e instanceof ApiError && e.status === 501) setNotice(e.message);
      else setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  }

  const upload = () =>
    run("Uploading", async () => {
      const res = await uploadTracks(files);
      setTracks(res.tracks);
      setStage("uploaded");
    });

  const analyze = () =>
    run("Analyzing", async () => {
      const res = await analyzeTracks(trackIds);
      setTracks(res.tracks);
      setStage("analyzed");
    });

  const order = () =>
    run("Ordering", async () => {
      const res = await orderTracks(trackIds);
      const byId = new Map(tracks.map((t) => [t.id, t]));
      setTracks(res.ordered_track_ids.map((id) => byId.get(id)!));
      setStage("ordered");
    });

  const mix = () =>
    run("Mixing", async () => {
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

  const currentStep = STEPS.findIndex((s) => s.stage === stage);

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 px-6 py-12">
      <header className="flex flex-col gap-2">
        <h1 className="text-4xl font-bold tracking-tight">Mixr</h1>
        <p className="text-neutral-400">Your playlist. DJ&apos;d by AI.</p>
      </header>

      <ol className="flex gap-2 text-sm">
        {STEPS.map((step, i) => (
          <li
            key={step.label}
            className={`flex-1 rounded-full px-3 py-1.5 text-center ${
              i < currentStep
                ? "bg-violet-500/20 text-violet-300"
                : i === currentStep
                  ? "bg-violet-500 text-white"
                  : "bg-neutral-800 text-neutral-500"
            }`}
          >
            {step.label}
          </li>
        ))}
      </ol>

      {stage === "select" && (
        <section className="flex flex-col gap-4">
          <label
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              pickFiles(e.dataTransfer.files);
            }}
            className={`flex cursor-pointer flex-col items-center gap-2 rounded-2xl border-2 border-dashed px-6 py-14 text-center transition ${
              dragging
                ? "border-violet-400 bg-violet-500/10"
                : "border-neutral-700 hover:border-neutral-500"
            }`}
          >
            <span className="text-lg font-medium">
              Drop {MIN_TRACKS}–{MAX_TRACKS} songs here
            </span>
            <span className="text-sm text-neutral-400">or click to choose MP3 or WAV files</span>
            <input
              ref={inputRef}
              type="file"
              multiple
              accept=".mp3,.wav,audio/mpeg,audio/wav"
              className="hidden"
              onChange={(e) => pickFiles(e.target.files)}
            />
          </label>

          {files.length > 0 && (
            <ul className="flex flex-col gap-1 text-sm text-neutral-300">
              {files.map((f) => (
                <li key={f.name + f.size} className="flex justify-between">
                  <span className="truncate">{f.name}</span>
                  <span className="text-neutral-500">{(f.size / 1024 / 1024).toFixed(1)} MB</span>
                </li>
              ))}
            </ul>
          )}

          {filesError && <p className="text-sm text-amber-400">{filesError}</p>}

          <button
            onClick={upload}
            disabled={files.length === 0 || !!filesError || !!busy}
            className="rounded-xl bg-violet-500 px-4 py-3 font-medium text-white transition hover:bg-violet-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ?? "Upload"}
          </button>
        </section>
      )}

      {stage === "mixed" && mixResult && <MixResult mix={mixResult} tracks={tracks} />}

      {stage !== "select" && (
        <section className="flex flex-col gap-4">
          {stage !== "mixed" && (
            <ul className="flex flex-col gap-3">
              {tracks.map((track, i) => (
                <li key={track.id} className="flex flex-col gap-3 rounded-2xl bg-neutral-900 p-4">
                  <div className="flex items-baseline justify-between gap-4">
                    <span className="truncate font-medium">
                      <span className="mr-2 text-neutral-500">{i + 1}.</span>
                      {track.filename}
                    </span>
                    <span className="shrink-0 text-sm text-neutral-400">
                      {formatSeconds(track.duration_seconds)}
                      {track.bpm != null && ` · ${Math.round(track.bpm)} BPM`}
                      {track.key_camelot && ` · ${track.key_camelot}`}
                    </span>
                  </div>
                  {track.analysis_status === "error" && (
                    <p className="text-sm text-red-400">{track.analysis_error}</p>
                  )}
                  <audio controls preload="none" src={trackAudioUrl(track.id)} className="w-full" />
                </li>
              ))}
            </ul>
          )}

          {stage === "analyzed" && tracks.every((t) => t.analysis_status === "pending") && (
            <p className="text-sm text-neutral-500">
              Audio analysis isn&apos;t connected yet, so BPM and key aren&apos;t shown.
            </p>
          )}

          <div className="flex gap-3">
            {stage === "uploaded" && <ActionButton label="Analyze" busy={busy} onClick={analyze} />}
            {stage === "analyzed" && <ActionButton label="Order" busy={busy} onClick={order} />}
            {stage === "ordered" && <ActionButton label="Mix" busy={busy} onClick={mix} />}
            <button
              onClick={reset}
              disabled={!!busy}
              className="rounded-xl border border-neutral-700 px-4 py-3 text-neutral-300 transition hover:border-neutral-500 disabled:opacity-40"
            >
              Start over
            </button>
          </div>
        </section>
      )}

      {notice && (
        <p className="rounded-xl bg-neutral-900 px-4 py-3 text-sm text-neutral-300">
          Not available yet: {notice}
        </p>
      )}
      {error && <p className="rounded-xl bg-red-500/10 px-4 py-3 text-sm text-red-300">{error}</p>}
    </div>
  );
}

function ActionButton({
  label,
  busy,
  onClick,
}: {
  label: string;
  busy: string | null;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={!!busy}
      className="flex-1 rounded-xl bg-violet-500 px-4 py-3 font-medium text-white transition hover:bg-violet-400 disabled:opacity-40"
    >
      {busy ?? label}
    </button>
  );
}

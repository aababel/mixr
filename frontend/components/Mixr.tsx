"use client";

import { useEffect, useRef, useState } from "react";

import {
  ApiError,
  FULL_SONGS,
  NEUTRAL_PREFERENCE,
  analyzeTracks,
  createMix,
  orderTracks,
  previewCalibration,
  uploadTracks,
  type CalibrationTrial,
  type MixOptions,
  type MixResult as Mix,
  type OrderExplanation as Explanation,
  type Preference,
  type Track,
} from "@/lib/api";

import Calibration from "./Calibration";
import Dropzone from "./Dropzone";
import MixResult from "./MixResult";
import ModePicker from "./ModePicker";
import OrderExplanation from "./OrderExplanation";
import Stepper from "./Stepper";
import TrackCard from "./TrackCard";
import Button from "./ui/Button";
import LoadingPanel from "./ui/LoadingPanel";

const MIN_TRACKS = 2;
const MAX_TRACKS = 5;
const ALLOWED_EXTENSIONS = [".mp3", ".wav"];
const MIN_LOADING_MS = 700;

type Stage = "select" | "uploaded" | "analyzed" | "ordered" | "tuning" | "mixed";
type Job = "upload" | "analyze" | "order" | "preview" | "mix";

const STEP_LABELS = ["Upload", "Analyze", "Order", "Tune", "Listen"];
const STEP_OF: Record<Stage, number> = {
  select: 0,
  uploaded: 1,
  analyzed: 2,
  ordered: 3,
  tuning: 3,
  mixed: 4,
};

const LOADING: Record<Job, { title: string; detail: string }> = {
  upload: { title: "Uploading your tracks", detail: "Making sure every file is real audio" },
  analyze: { title: "Listening closely", detail: "Reading tempo, beats, key and energy" },
  order: { title: "Finding the best order", detail: "Scoring how each song flows into the next" },
  preview: { title: "Rendering two previews", detail: "Same transition, two overlap lengths" },
  mix: { title: "Mixing", detail: "Beat-matching and blending every transition" },
};

const HEADINGS: Record<Exclude<Stage, "tuning" | "mixed">, { title: string; detail: string }> = {
  select: {
    title: "Start a new mix",
    detail: `Pick ${MIN_TRACKS} to ${MAX_TRACKS} songs you want blended together.`,
  },
  uploaded: { title: "Your tracks", detail: "Next, Mixr analyzes each song." },
  analyzed: {
    title: "Analyzed",
    detail: "Next, Mixr finds the order that flows best, or drag the songs into your own.",
  },
  ordered: {
    title: "Your running order",
    detail:
      "Drag to rearrange if you want, then tune the transitions in a few quick comparisons or mix right away.",
  },
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

const TYPICAL_BLEND_SECONDS = 8;

/** Rough length of a full-songs mix, before the real blends are rendered. */
const fullMixSeconds = (tracks: Track[]) =>
  tracks.reduce((sum, t) => sum + t.duration_seconds, 0) -
  Math.max(0, tracks.length - 1) * TYPICAL_BLEND_SECONDS;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export default function Mixr() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [tracks, setTracks] = useState<Track[]>([]);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [explainedOrder, setExplainedOrder] = useState<string | null>(null);
  const [suggestedIds, setSuggestedIds] = useState<string[] | null>(null);
  const [dragFrom, setDragFrom] = useState<number | null>(null);
  const [dragOver, setDragOver] = useState<number | null>(null);
  const [mixResult, setMixResult] = useState<Mix | null>(null);
  const [preference, setPreference] = useState<Preference>(NEUTRAL_PREFERENCE);
  const [firstTrial, setFirstTrial] = useState<CalibrationTrial | null>(null);
  const [mixOptions, setMixOptions] = useState<MixOptions>(FULL_SONGS);
  const [stage, setStage] = useState<Stage>("select");
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ label: string; text: string } | null>(null);

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
      setError(e instanceof Error ? e.message : String(e));
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

  const order = (keepMine = false) =>
    run("order", async () => {
      const res = await orderTracks(trackIds, keepMine);
      const byId = new Map(tracks.map((t) => [t.id, t]));
      setTracks(res.ordered_track_ids.map((id) => byId.get(id)!));
      setSuggestedIds(res.suggested_track_ids);
      setExplanation(res.explanation);
      setExplainedOrder(res.ordered_track_ids.join());
      setStage("ordered");
    });

  // Re-explain the order in place (no loading panel) whenever the user rearranges it.
  const orderKey = trackIds.join();
  useEffect(() => {
    if (stage !== "ordered" || explainedOrder === null || explainedOrder === orderKey) return;
    let stale = false;
    orderTracks(orderKey.split(","), true)
      .then((res) => {
        if (stale) return;
        setExplanation(res.explanation);
        setExplainedOrder(orderKey);
      })
      .catch((e) => {
        if (!stale) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      stale = true;
    };
  }, [stage, orderKey, explainedOrder]);

  function moveTrack(from: number, to: number) {
    if (from === to || to < 0 || to >= tracks.length) return;
    const next = [...tracks];
    const [moved] = next.splice(from, 1);
    next.splice(to, 0, moved);
    setTracks(next);
  }

  function restoreMixrOrder() {
    if (!suggestedIds) return;
    const byId = new Map(tracks.map((t) => [t.id, t]));
    setTracks(suggestedIds.map((id) => byId.get(id)!));
  }

  const canReorder = (stage === "analyzed" || stage === "ordered") && !job;
  const isCustomOrder =
    stage === "ordered" && suggestedIds !== null && suggestedIds.join() !== trackIds.join();

  const tune = () =>
    run("preview", async () => {
      try {
        setFirstTrial(await previewCalibration(trackIds, 0, [], mixOptions));
        setStage("tuning");
      } catch (e) {
        if (!(e instanceof ApiError && e.status === 422)) throw e;
        setJob("mix");
        setMixResult(await createMix(trackIds, preference, mixOptions));
        setStage("mixed");
        setNotice({
          label: "Skipped tuning",
          text: "blend length wouldn't make an audible difference for these songs, so Mixr used its default transitions.",
        });
      }
    });

  const mix = () =>
    run("mix", async () => {
      setMixResult(await createMix(trackIds, preference, mixOptions));
      setStage("mixed");
    });

  function reset() {
    setFiles([]);
    setTracks([]);
    setExplanation(null);
    setExplainedOrder(null);
    setSuggestedIds(null);
    setMixResult(null);
    setPreference(NEUTRAL_PREFERENCE);
    setMixOptions(FULL_SONGS);
    setFirstTrial(null);
    setStage("select");
    setError(null);
    setNotice(null);
    if (inputRef.current) inputRef.current.value = "";
  }

  const nextAction = {
    uploaded: { label: "Analyze tracks", onClick: analyze },
    analyzed: { label: "Find the best order", onClick: () => order() },
    ordered: { label: "Tune my transitions", onClick: tune },
  }[stage as "uploaded" | "analyzed" | "ordered"];

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-8 px-6 pt-4 pb-16">
      <Stepper steps={STEP_LABELS} current={STEP_OF[stage]} />

      {stage !== "mixed" && stage !== "tuning" && !job && (
        <div className="animate-fade-up flex flex-col gap-1">
          <h1 className="font-display text-3xl font-semibold">{HEADINGS[stage].title}</h1>
          <p className="text-neutral-400">{HEADINGS[stage].detail}</p>
        </div>
      )}

      {job ? (
        <LoadingPanel {...LOADING[job]} />
      ) : stage === "tuning" && firstTrial ? (
        <Calibration
          orderedIds={trackIds}
          tracks={tracks}
          initialTrial={firstTrial}
          preference={preference}
          options={mixOptions}
          onPreference={setPreference}
          onMix={mix}
        />
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
          {stage === "ordered" && explanation && (
            <OrderExplanation
              explanation={explanation}
              tracks={tracks}
              onReset={isCustomOrder ? restoreMixrOrder : undefined}
            />
          )}
          {stage === "mixed" && mixResult ? (
            <MixResult mix={mixResult} tracks={tracks} />
          ) : (
            <ul className="flex flex-col gap-3">
              {tracks.map((track, i) => (
                <TrackCard
                  key={track.id}
                  track={track}
                  position={i + 1}
                  reorder={
                    canReorder
                      ? {
                          canUp: i > 0,
                          canDown: i < tracks.length - 1,
                          onMove: (delta) => moveTrack(i, i + delta),
                          onDragStart: () => setDragFrom(i),
                          onDragEnter: () => setDragOver(i),
                          onDrop: () => {
                            if (dragFrom !== null) moveTrack(dragFrom, i);
                            setDragFrom(null);
                            setDragOver(null);
                          },
                          onDragEnd: () => {
                            setDragFrom(null);
                            setDragOver(null);
                          },
                          dragging: dragFrom === i,
                          dropTarget: dragFrom !== null && dragOver === i && dragFrom !== i,
                        }
                      : undefined
                  }
                />
              ))}
            </ul>
          )}

          {stage === "ordered" && (
            <ModePicker
              songs={tracks.length}
              fullSeconds={fullMixSeconds(tracks)}
              options={mixOptions}
              onChange={setMixOptions}
            />
          )}

          <div className="flex flex-col gap-3 sm:flex-row">
            {nextAction && (
              <Button onClick={nextAction.onClick} className="flex-1">
                {nextAction.label}
              </Button>
            )}
            {stage === "analyzed" && (
              <Button variant="ghost" onClick={() => order(true)}>
                Use my order
              </Button>
            )}
            {stage === "ordered" && (
              <Button variant="ghost" onClick={mix}>
                Skip, mix now
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
          <span className="text-neon font-semibold">{notice.label}:</span> {notice.text}
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

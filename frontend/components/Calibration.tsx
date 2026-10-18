"use client";

import { useState } from "react";

import {
  ApiError,
  apiUrl,
  previewCalibration,
  voteCalibration,
  type CalibrationTrial,
  type MixOptions,
  type Preference,
  type Track,
  type VoteResult,
} from "@/lib/api";

import Button from "./ui/Button";
import Equalizer from "./ui/Equalizer";

const MAX_TRIALS = 3;

export default function Calibration({
  orderedIds,
  tracks,
  initialTrial,
  preference,
  options,
  onPreference,
  onMix,
}: {
  orderedIds: string[];
  tracks: Track[];
  initialTrial: CalibrationTrial;
  preference: Preference;
  options: MixOptions;
  onPreference: (preference: Preference) => void;
  onMix: () => void;
}) {
  const [trial, setTrial] = useState(initialTrial);
  const [trialIndex, setTrialIndex] = useState(0);
  const [compared, setCompared] = useState<[string, string][]>([
    [initialTrial.from_id, initialTrial.to_id],
  ]);
  const [exhausted, setExhausted] = useState(false);
  const [result, setResult] = useState<VoteResult | null>(null);
  const [chosen, setChosen] = useState<"A" | "B" | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const name = (id: string) => tracks.find((t) => t.id === id)?.filename ?? id;
  const votes = preference.votes.length;
  // One comparison per pair of neighboring songs, never the same pair twice.
  const totalTrials = Math.min(MAX_TRIALS, orderedIds.length - 1);
  const canContinue = !!result && votes < totalTrials && !error && !exhausted;

  async function attempt(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const vote = (variant: "A" | "B") =>
    attempt(async () => {
      const res = await voteCalibration(trial.trial_id, variant, preference);
      setResult(res);
      setChosen(variant);
      onPreference(res.preference);
    });

  const nextTrial = () =>
    attempt(async () => {
      let next: CalibrationTrial;
      try {
        next = await previewCalibration(orderedIds, trialIndex + 1, compared, options);
      } catch (e) {
        if (e instanceof ApiError && e.status === 422) {
          setExhausted(true);
          return;
        }
        throw e;
      }
      setTrial(next);
      setTrialIndex(trialIndex + 1);
      setCompared([...compared, [next.from_id, next.to_id]]);
      setResult(null);
      setChosen(null);
    });

  return (
    <section className="animate-fade-up flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <p className="font-mono text-xs tracking-widest text-neutral-500 uppercase">
          Comparison {trialIndex + 1} of {totalTrials}
        </p>
        <h1 className="font-display text-3xl font-semibold">Which blend do you like?</h1>
        <p className="text-neutral-400">
          Same two songs. Only how long they overlap changes. Listen to both, then pick.
        </p>
      </div>

      <p className="glass truncate rounded-2xl px-4 py-3 text-sm">
        {name(trial.from_id)} <span className="text-neon font-semibold">→</span> {name(trial.to_id)}
      </p>

      <div key={trial.trial_id} className="grid gap-4 sm:grid-cols-2">
        {(["A", "B"] as const).map((variant) => {
          const picked = chosen === variant;
          return (
            <div
              key={variant}
              className={`glass flex flex-col gap-4 rounded-3xl p-5 transition ${
                picked ? "glow border-neon-pink/60" : ""
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-display text-4xl font-semibold">{variant}</span>
                {picked && <Equalizer bars={4} className="h-6" barClassName="w-1" />}
              </div>
              <audio
                controls
                preload="auto"
                src={apiUrl(variant === "A" ? trial.a.audio_url : trial.b.audio_url)}
                className="w-full"
              />
              {!result && (
                <Button onClick={() => vote(variant)} disabled={busy}>
                  Pick {variant}
                </Button>
              )}
            </div>
          );
        })}
      </div>

      {result && (
        <div className="glass animate-fade-up flex flex-col gap-4 rounded-3xl p-5">
          <p>
            You picked the{" "}
            <span className="text-neon font-semibold">
              {result.chose_longer ? "longer" : "shorter"} blend
            </span>{" "}
            ({result.chosen_overlap_seconds.toFixed(1)}s vs{" "}
            {result.other_overlap_seconds.toFixed(1)}s).
          </p>
          <PreferenceMeter value={preference.overlap_preference} />
        </div>
      )}

      {exhausted && (
        <p className="text-sm text-neutral-400">
          That&apos;s every pair of songs where blend length makes an audible difference.
        </p>
      )}

      {error && (
        <p className="rounded-2xl border border-red-400/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </p>
      )}

      <div className="flex flex-col gap-3 sm:flex-row">
        {canContinue && (
          <Button onClick={nextTrial} disabled={busy} className="flex-1">
            {busy ? "Rendering previews…" : "Next comparison"}
          </Button>
        )}
        <Button
          variant={result && !canContinue ? "primary" : "ghost"}
          onClick={onMix}
          disabled={busy}
          className={canContinue ? "" : "flex-1"}
        >
          {votes > 0 ? "Mix it with my taste" : "Skip, use default transitions"}
        </Button>
      </div>
    </section>
  );
}

function PreferenceMeter({ value }: { value: number }) {
  const percent = ((value + 1) / 2) * 100;
  return (
    <div className="flex flex-col gap-2 text-xs text-neutral-400">
      <div className="relative h-2 rounded-full bg-white/10">
        <div
          className="bg-neon glow absolute top-1/2 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full transition-all duration-500"
          style={{ left: `${percent}%` }}
        />
      </div>
      <div className="flex justify-between font-mono tracking-wide uppercase">
        <span>Quick cuts</span>
        <span>Long blends</span>
      </div>
    </div>
  );
}

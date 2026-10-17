"use client";

import { useRef } from "react";

import { mixAudioUrl, type MixResult as Mix, type Track } from "@/lib/api";
import { formatSeconds } from "@/lib/format";

export default function MixResult({ mix, tracks }: { mix: Mix; tracks: Track[] }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const { manifest } = mix;
  const byId = new Map(tracks.map((t) => [t.id, t]));
  const name = (id: string) => byId.get(id)?.filename ?? id;

  const starts = manifest.ordered_track_ids.reduce<number[]>((acc, _id, i) => {
    if (i === 0) return [0];
    const seam = manifest.transitions[i - 1];
    return [...acc, acc[i - 1] + (seam?.outgoing_start_seconds ?? 0)];
  }, []);

  function seek(seconds: number) {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = seconds;
    audio.play();
  }

  return (
    <section className="flex flex-col gap-6 rounded-2xl bg-neutral-900 p-5">
      <div className="flex items-baseline justify-between gap-4">
        <h2 className="text-2xl font-semibold">Your mix</h2>
        <span className="text-sm text-neutral-400">
          {formatSeconds(manifest.output_duration_seconds)} ·{" "}
          {manifest.calibrated ? "tuned to your taste" : "default transitions"}
        </span>
      </div>

      <audio ref={audioRef} controls src={mixAudioUrl(mix)} className="w-full" />

      <a
        href={mixAudioUrl(mix)}
        className="rounded-xl bg-violet-500 px-4 py-3 text-center font-medium text-white transition hover:bg-violet-400"
      >
        Download mix (WAV)
      </a>

      <div className="flex flex-col gap-2">
        <h3 className="text-sm font-medium uppercase tracking-wide text-neutral-500">Tracklist</h3>
        <ol className="flex flex-col gap-1 text-sm">
          {manifest.ordered_track_ids.map((id, i) => (
            <li key={id}>
              <button
                onClick={() => seek(starts[i])}
                className="flex w-full justify-between gap-4 rounded-lg px-2 py-1.5 text-left hover:bg-neutral-800"
              >
                <span className="truncate">
                  <span className="mr-2 text-neutral-500">{i + 1}.</span>
                  {name(id)}
                </span>
                <span className="shrink-0 font-mono text-neutral-400">
                  {formatSeconds(starts[i])}
                </span>
              </button>
            </li>
          ))}
        </ol>
      </div>

      {manifest.transitions.length > 0 && (
        <div className="flex flex-col gap-2">
          <h3 className="text-sm font-medium uppercase tracking-wide text-neutral-500">
            Transitions
          </h3>
          <ul className="flex flex-col gap-2 text-sm">
            {manifest.transitions.map((t) => (
              <li
                key={`${t.from_id}-${t.to_id}`}
                className="rounded-lg bg-neutral-800/60 px-3 py-2"
              >
                <div className="truncate">
                  {name(t.from_id)} <span className="text-neutral-500">→</span> {name(t.to_id)}
                </div>
                <div className="text-neutral-400">
                  {t.actual_overlap_seconds.toFixed(1)}s {t.fade_style.replace("_", "-")} overlap
                  {t.overlap_changed_for_safety &&
                    ` (asked for ${t.requested_overlap_seconds.toFixed(1)}s, shortened for safety)`}
                  {t.beat_alignment_adjustments_seconds !== 0 &&
                    ` · beat-aligned by ${t.beat_alignment_adjustments_seconds.toFixed(2)}s`}
                  {t.tempo_change != null && ` · tempo ×${t.tempo_change.toFixed(2)}`}
                  {t.applied_eq && ` · EQ: ${t.applied_eq}`}
                </div>
                {t.safety_adjustments.length > 0 && (
                  <ul className="mt-1 list-disc pl-5 text-amber-300/80">
                    {t.safety_adjustments.map((s) => (
                      <li key={s}>{s}</li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {manifest.limitations.length > 0 && (
        <div className="flex flex-col gap-1 text-sm text-neutral-500">
          <h3 className="font-medium uppercase tracking-wide">Limitations</h3>
          <ul className="list-disc pl-5">
            {manifest.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

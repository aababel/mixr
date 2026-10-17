"use client";

import { useRef, useState } from "react";

import { mixAudioUrl, type MixResult as Mix, type Track } from "@/lib/api";
import { formatSeconds } from "@/lib/format";

import { buttonClasses } from "./ui/Button";
import Equalizer from "./ui/Equalizer";
import Vinyl from "./ui/Vinyl";

export default function MixResult({ mix, tracks }: { mix: Mix; tracks: Track[] }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
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
    <div className="animate-fade-up flex flex-col gap-6">
      <section className="glass flex flex-col gap-6 rounded-3xl p-6">
        <div className="flex items-center gap-5">
          <Vinyl className="size-20 shrink-0" spinning={playing} />
          <div className="flex min-w-0 flex-col gap-1">
            <span className="text-xs tracking-widest text-neutral-400 uppercase">Your mix</span>
            <h2 className="font-display text-neon text-3xl font-bold">Ready to play</h2>
            <span className="text-sm text-neutral-400">
              {manifest.ordered_track_ids.length} songs ·{" "}
              {formatSeconds(manifest.output_duration_seconds)} ·{" "}
              {manifest.calibrated ? "tuned to your taste" : "default transitions"}
            </span>
          </div>
          {playing && <Equalizer bars={5} className="ml-auto hidden h-8 sm:inline-flex" />}
        </div>

        <audio
          ref={audioRef}
          controls
          src={mixAudioUrl(mix)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          className="w-full"
        />

        <a href={mixAudioUrl(mix)} className={buttonClasses("primary")}>
          Download mix (WAV)
        </a>
      </section>

      <section className="glass flex flex-col gap-3 rounded-3xl p-6">
        <h3 className="text-xs tracking-widest text-neutral-400 uppercase">Tracklist</h3>
        <ol className="flex flex-col">
          {manifest.ordered_track_ids.map((id, i) => (
            <li key={id}>
              <button
                onClick={() => seek(starts[i])}
                className="group flex w-full items-center gap-4 rounded-xl px-2 py-2.5 text-left transition hover:bg-white/5"
              >
                <span className="font-display text-neon w-7 text-lg font-bold">
                  {(i + 1).toString().padStart(2, "0")}
                </span>
                <span className="flex-1 truncate">{name(id)}</span>
                <span className="font-mono text-sm text-neutral-400 group-hover:text-white">
                  {formatSeconds(starts[i])}
                </span>
              </button>
            </li>
          ))}
        </ol>
      </section>

      {manifest.transitions.length > 0 && (
        <section className="glass flex flex-col gap-3 rounded-3xl p-6">
          <h3 className="text-xs tracking-widest text-neutral-400 uppercase">Transitions</h3>
          <ul className="flex flex-col gap-2 text-sm">
            {manifest.transitions.map((t) => (
              <li
                key={`${t.from_id}-${t.to_id}`}
                className="flex flex-col gap-1 rounded-xl bg-white/[0.03] px-4 py-3"
              >
                <div className="flex items-center gap-2">
                  <span className="truncate">{name(t.from_id)}</span>
                  <span className="text-neon shrink-0 font-bold">→</span>
                  <span className="truncate">{name(t.to_id)}</span>
                </div>
                <div className="text-neutral-400">
                  {t.actual_overlap_seconds.toFixed(1)}s {t.fade_style.replace("_", "-")} blend
                  {t.overlap_changed_for_safety &&
                    ` (asked for ${t.requested_overlap_seconds.toFixed(1)}s, shortened for safety)`}
                  {t.beat_alignment_adjustments_seconds !== 0 &&
                    ` · beat-aligned by ${t.beat_alignment_adjustments_seconds.toFixed(2)}s`}
                  {t.tempo_change != null && ` · tempo ×${t.tempo_change.toFixed(2)}`}
                  {t.applied_eq && ` · EQ: ${t.applied_eq}`}
                </div>
                {t.safety_adjustments.length > 0 && (
                  <ul className="list-disc pl-5 text-amber-300/80">
                    {t.safety_adjustments.map((s) => (
                      <li key={s}>{s}</li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}

      {manifest.limitations.length > 0 && (
        <section className="flex flex-col gap-1 px-2 text-sm text-neutral-500">
          <h3 className="text-xs tracking-widest uppercase">Limitations</h3>
          <ul className="list-disc pl-5">
            {manifest.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

"use client";

import type { KeyboardEvent, MouseEvent } from "react";

import type { Track } from "@/lib/api";
import { formatSeconds } from "@/lib/format";
import {
  BLEND_PREROLL_SECONDS,
  energyPath,
  playedEnergy,
  trackColor,
  type Blend,
  type Segment,
} from "@/lib/timeline";

const WAVE_WIDTH = 120;
const SEEK_STEP_SECONDS = 5;

type Props = {
  total: number;
  segments: Segment[];
  blends: Blend[];
  tracks: Map<string, Track>;
  currentTime: number;
  onSeek: (seconds: number) => void;
};

const pct = (seconds: number, total: number) => `${(seconds / total) * 100}%`;

export default function MixTimeline({ total, segments, blends, tracks, currentTime, onSeek }: Props) {
  const peak = Math.max(0, ...segments.flatMap((s) => playedEnergy(tracks.get(s.id), s.window)));
  const clamp = (s: number) => Math.min(total, Math.max(0, s));

  function seekToPointer(e: MouseEvent<HTMLDivElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    onSeek(clamp(((e.clientX - rect.left) / rect.width) * total));
  }

  function seekWithKeys(e: KeyboardEvent<HTMLDivElement>) {
    const step = { ArrowLeft: -SEEK_STEP_SECONDS, ArrowRight: SEEK_STEP_SECONDS }[e.key];
    if (step === undefined) return;
    e.preventDefault();
    onSeek(clamp(currentTime + step));
  }

  if (total <= 0) return null;

  return (
    <div className="flex flex-col gap-2">
      <div
        role="slider"
        tabIndex={0}
        aria-label="Mix timeline"
        aria-valuemin={0}
        aria-valuemax={Math.round(total)}
        aria-valuenow={Math.round(currentTime)}
        aria-valuetext={formatSeconds(currentTime)}
        onClick={seekToPointer}
        onKeyDown={seekWithKeys}
        className="relative h-32 cursor-pointer rounded-2xl bg-white/[0.03] outline-none focus-visible:ring-2 focus-visible:ring-neon-pink/60"
      >
        {segments.map((s) => {
          const track = tracks.get(s.id);
          const color = trackColor(s.index);
          return (
            <div
              key={s.id}
              className="absolute flex flex-col overflow-hidden rounded-xl border"
              style={{
                left: pct(s.start, total),
                width: pct(s.end - s.start, total),
                top: s.index % 2 === 0 ? "8%" : "52%",
                height: "40%",
                borderColor: `${color}66`,
                background: `linear-gradient(180deg, ${color}26, ${color}0d)`,
              }}
            >
              <span
                className="absolute top-1 left-2 z-10 max-w-[calc(100%-1rem)] truncate text-[10px] font-semibold tracking-wide"
                style={{ color }}
              >
                {(s.index + 1).toString().padStart(2, "0")} · {track?.filename ?? s.id}
              </span>
              <svg
                viewBox={`0 0 ${WAVE_WIDTH} 100`}
                preserveAspectRatio="none"
                className="mt-auto h-3/4 w-full"
                aria-hidden
              >
                <path d={energyPath(playedEnergy(track, s.window), peak, WAVE_WIDTH)} fill={color} fillOpacity={0.55} />
              </svg>
            </div>
          );
        })}

        {blends.map((b) => (
          <button
            key={b.index}
            type="button"
            title={`Play blend ${b.index + 1}`}
            onClick={(e) => {
              e.stopPropagation();
              onSeek(clamp(b.start - BLEND_PREROLL_SECONDS));
            }}
            className="group absolute top-0 bottom-0 z-20 flex min-w-3 items-end justify-center rounded-lg border-x border-white/40 bg-white/10 pb-1 transition hover:bg-white/20"
            style={{ left: pct(b.start, total), width: pct(b.end - b.start, total) }}
          >
            <span className="rounded-full bg-night/80 px-1.5 text-[10px] font-bold text-white">
              {b.index + 1}
            </span>
          </button>
        ))}

        <div
          className="pointer-events-none absolute top-0 bottom-0 z-30 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_12px_2px_rgba(255,43,214,0.7)]"
          style={{ left: pct(clamp(currentTime), total) }}
        />
      </div>

      <div className="flex justify-between font-mono text-xs text-neutral-500">
        <span>0:00</span>
        <span className="text-neutral-300">{formatSeconds(currentTime)}</span>
        <span>{formatSeconds(total)}</span>
      </div>
    </div>
  );
}

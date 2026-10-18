import type { AppliedTransition, MixManifest, Track, TrackWindow } from "./api";

export const TRACK_COLORS = ["#ff2bd6", "#8b5cf6", "#22d3ee", "#f59e0b", "#34d399"];

/** Start blend playback this many seconds early so you hear the lead-in. */
export const BLEND_PREROLL_SECONDS = 4;

export const trackColor = (index: number) => TRACK_COLORS[index % TRACK_COLORS.length];

/** window: the part of the source song this segment plays (null = whole song). */
export type Segment = { id: string; index: number; start: number; end: number; window: TrackWindow | null };
export type Blend = { index: number; start: number; end: number; transition: AppliedTransition };

/** Where each track and each blend sits on the mix timeline, in seconds. */
export function mixLayout(manifest: MixManifest) {
  const total = manifest.output_duration_seconds;
  const { ordered_track_ids: ids, transitions, source_durations_seconds: durations } = manifest;
  const segments: Segment[] = [];
  const blends: Blend[] = [];

  let start = 0;
  // A tempo glide stretches the head of the incoming track: its first
  // actual_overlap_seconds of source audio last output_duration_seconds in the mix.
  let headStretch = 0;
  ids.forEach((id, i) => {
    const next = transitions[i];
    const played = manifest.windows?.[i] ?? null;
    if (!next) {
      const end = Math.min(total, start + (durations[i] ?? 0) + headStretch);
      segments.push({ id, index: i, start, end, window: played });
      return;
    }
    // outgoing_start_seconds is measured from the start of the outgoing window
    // (the track itself without windows), and durations are window lengths.
    const blendStart = start + next.outgoing_start_seconds + headStretch;
    const blendEnd = Math.min(total, blendStart + next.output_duration_seconds);
    segments.push({ id, index: i, start, end: blendEnd, window: played });
    blends.push({ index: i, start: blendStart, end: blendEnd, transition: next });
    start = blendStart;
    headStretch = next.output_duration_seconds - next.actual_overlap_seconds;
  });

  return { total, segments, blends };
}

/** True when a window cuts something off the song (not just float noise at its edges). */
export function isTrimmed(window: TrackWindow | null, duration: number | undefined) {
  if (!window) return false;
  return window.start_seconds > 0.5 || (duration !== undefined && window.end_seconds < duration - 0.5);
}

/** The part of a track's RMS envelope its window plays. */
export function playedEnergy(track: Track | undefined, window: TrackWindow | null) {
  const energy = track?.energy_rms ?? [];
  const hop = track?.energy_hop_seconds;
  if (!window || !hop) return energy;
  return energy.slice(Math.floor(window.start_seconds / hop), Math.ceil(window.end_seconds / hop));
}

/** Index of the track that is "on" at a given time (the incoming one during a blend). */
export function activeIndex(segments: Segment[], seconds: number) {
  let active = 0;
  segments.forEach((s, i) => {
    if (seconds >= s.start) active = i;
  });
  return active;
}

/** Mirrored waveform path in a 0..width x 0..100 box from an RMS envelope. */
export function energyPath(energy: number[], peak: number, width = 120) {
  if (energy.length === 0 || peak <= 0) return "";
  const buckets = Math.min(width, energy.length);
  const heights = Array.from({ length: buckets }, (_, b) => {
    const from = Math.floor((b * energy.length) / buckets);
    const to = Math.max(from + 1, Math.floor(((b + 1) * energy.length) / buckets));
    return Math.max(...energy.slice(from, to)) / peak;
  });
  const x = (b: number) => ((b + 0.5) / buckets) * width;
  const top = heights.map((h, b) => `${x(b).toFixed(2)},${(50 - h * 46).toFixed(2)}`);
  const bottom = heights.map((h, b) => `${x(b).toFixed(2)},${(50 + h * 46).toFixed(2)}`).reverse();
  return `M0,50 L${top.join(" L")} L${width},50 L${bottom.join(" L")} Z`;
}

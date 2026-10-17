import type { AppliedTransition, MixManifest } from "./api";

export const TRACK_COLORS = ["#ff2bd6", "#8b5cf6", "#22d3ee", "#f59e0b", "#34d399"];

/** Start blend playback this many seconds early so you hear the lead-in. */
export const BLEND_PREROLL_SECONDS = 4;

export const trackColor = (index: number) => TRACK_COLORS[index % TRACK_COLORS.length];

export type Segment = { id: string; index: number; start: number; end: number };
export type Blend = { index: number; start: number; end: number; transition: AppliedTransition };

/** Where each track and each blend sits on the mix timeline, in seconds. */
export function mixLayout(manifest: MixManifest) {
  const total = manifest.output_duration_seconds;
  const segments: Segment[] = [];
  const blends: Blend[] = [];

  manifest.ordered_track_ids.forEach((id, i) => {
    const previous = manifest.transitions[i - 1];
    // outgoing_start_seconds is measured from the start of the outgoing track.
    const start = i === 0 ? 0 : segments[i - 1].start + (previous?.outgoing_start_seconds ?? 0);
    const end = Math.min(total, start + (manifest.source_durations_seconds[i] ?? 0));
    segments.push({ id, index: i, start, end });
    if (previous) {
      blends.push({
        index: i - 1,
        start,
        end: Math.min(total, start + previous.output_duration_seconds),
        transition: previous,
      });
    }
  });

  return { total, segments, blends };
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

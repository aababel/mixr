export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Track = {
  id: string;
  filename: string;
  duration_seconds: number;
  sample_rate: number;
  bpm: number | null;
  beats_seconds: number[];
  key_camelot: string | null;
  energy_rms: number[];
  energy_hop_seconds: number | null;
  analysis_status: "pending" | "ok" | "error";
  analysis_error: string | null;
  metadata: Record<string, unknown>;
};

export type PairScore = {
  from_id: string;
  to_id: string;
  total: number;
  harmonic: number | null;
  tempo: number | null;
  energy: number | null;
  mood: number | null;
};

export type TransitionExplanation = {
  from_id: string;
  to_id: string;
  reasons: string[];
  score: number;
};

export type OrderExplanation = {
  summary: string;
  transitions: TransitionExplanation[];
};

export type AppliedTransition = {
  from_id: string;
  to_id: string;
  mode: "preview" | "final";
  fade_style: "equal_power" | "linear";
  requested_overlap_seconds: number;
  actual_overlap_seconds: number;
  overlap_changed_for_safety: boolean;
  outgoing_start_seconds: number;
  incoming_start_seconds: number;
  beat_alignment_adjustments_seconds: number;
  applied_gain_db: Record<string, number>;
  applied_eq: string | null;
  tempo_change: number | null;
  safety_adjustments: string[];
  output_duration_seconds: number;
};

/** The part of a song the mix plays, in seconds of the source song. */
export type TrackWindow = {
  track_id: string;
  start_seconds: number;
  end_seconds: number;
  reason: string | null;
  /** Set when the song enters with a drop transition: its drop lands as the previous song leaves. */
  drop_seconds?: number | null;
};

export type MixManifest = {
  mix_id: string;
  ordered_track_ids: string[];
  transitions: AppliedTransition[];
  /** Window lengths: transition times are measured from each window's start. */
  source_durations_seconds: number[];
  output_duration_seconds: number;
  calibrated: boolean;
  overlap_preference: number;
  limitations: string[];
  mode: "full" | "set";
  /** Same order as ordered_track_ids; empty for mixes made without cue points. */
  windows: TrackWindow[];
};

export type MixResult = {
  mix_id: string;
  audio_url: string;
  manifest: MixManifest;
};

export type Vote = {
  from_id: string;
  to_id: string;
  a_requested_seconds: number;
  b_requested_seconds: number;
  chosen: "A" | "B";
};

export type Preference = {
  overlap_preference: number;
  votes: Vote[];
  completed: boolean;
};

export const NEUTRAL_PREFERENCE: Preference = {
  overlap_preference: 0,
  votes: [],
  completed: false,
};

export type CalibrationTrial = {
  trial_id: string;
  from_id: string;
  to_id: string;
  a: { audio_url: string };
  b: { audio_url: string };
};

export type VoteResult = {
  preference: Preference;
  chosen_overlap_seconds: number;
  other_overlap_seconds: number;
  chose_longer: boolean;
};

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, init);
  } catch {
    throw new ApiError(`Can't reach the Mixr backend at ${API_URL}. Is it running?`, 0);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : res.statusText;
    throw new ApiError(detail, res.status);
  }
  return res.json();
}

function postJson<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function uploadTracks(files: File[]) {
  const form = new FormData();
  files.forEach((file) => form.append("files", file));
  return request<{ track_ids: string[]; tracks: Track[] }>("/api/tracks", {
    method: "POST",
    body: form,
  });
}

export function analyzeTracks(trackIds: string[]) {
  return postJson<{ tracks: Track[] }>("/api/analyze", { track_ids: trackIds });
}

/** keepOrder: explain trackIds as given instead of reordering them. */
export function orderTracks(trackIds: string[], keepOrder = false) {
  return postJson<{
    ordered_track_ids: string[];
    suggested_track_ids: string[];
    scores: PairScore[];
    explanation: OrderExplanation;
  }>("/api/order", { track_ids: trackIds, keep_order: keepOrder });
}

/** "auto" sizes each blend to its pair; "longer" leans every blend longer;
 * "extended" allows blends up to about a minute (96-128 beats) where the songs suit it. */
export type TransitionLength = "auto" | "longer" | "extended";

/** "full" plays every song; "set" trims each song to its best part to hit a target length. */
export type MixOptions = {
  mode: "full" | "set";
  targetSeconds: number | null;
  transitionLength: TransitionLength;
};

export const FULL_SONGS: MixOptions = { mode: "full", targetSeconds: null, transitionLength: "auto" };

const modeFields = (options: MixOptions) => ({
  mode: options.mode,
  target_seconds: options.mode === "set" ? options.targetSeconds : null,
  transition_length: options.transitionLength,
});

export function createMix(
  orderedTrackIds: string[],
  preference: Preference = NEUTRAL_PREFERENCE,
  options: MixOptions = FULL_SONGS,
) {
  return postJson<MixResult>("/api/mix", {
    ordered_track_ids: orderedTrackIds,
    overlap_preference: preference.overlap_preference,
    calibrated: preference.votes.length > 0,
    ...modeFields(options),
  });
}

export function previewCalibration(
  orderedTrackIds: string[],
  trial: number,
  excludePairs: [string, string][] = [],
  options: MixOptions = FULL_SONGS,
) {
  return postJson<CalibrationTrial>("/api/calibration/preview", {
    ordered_track_ids: orderedTrackIds,
    trial,
    exclude_pairs: excludePairs,
    ...modeFields(options),
  });
}

export function voteCalibration(trialId: string, chosen: "A" | "B", preference: Preference) {
  return postJson<VoteResult>("/api/calibration/vote", {
    trial_id: trialId,
    chosen,
    preference,
  });
}

export function apiUrl(path: string) {
  return `${API_URL}${path}`;
}

export function trackAudioUrl(trackId: string) {
  return `${API_URL}/api/tracks/${trackId}/audio`;
}

export function mixAudioUrl(mix: MixResult) {
  return `${API_URL}${mix.audio_url}`;
}

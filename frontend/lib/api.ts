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

export function orderTracks(trackIds: string[]) {
  return postJson<{ ordered_track_ids: string[]; scores: PairScore[] }>("/api/order", {
    track_ids: trackIds,
  });
}

export function createMix(orderedTrackIds: string[]) {
  return postJson<{ mix_id: string }>("/api/mix", { ordered_track_ids: orderedTrackIds });
}

export function trackAudioUrl(trackId: string) {
  return `${API_URL}/api/tracks/${trackId}/audio`;
}

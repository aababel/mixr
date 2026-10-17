import { trackAudioUrl, type Track } from "@/lib/api";
import { formatSeconds } from "@/lib/format";

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full bg-white/8 px-2.5 py-0.5 font-mono text-xs text-neutral-300">
      {children}
    </span>
  );
}

export default function TrackCard({ track, position }: { track: Track; position: number }) {
  return (
    <li className="glass animate-fade-up flex flex-col gap-3 rounded-2xl p-4">
      <div className="flex items-center gap-4">
        <span className="font-display text-neon w-12 shrink-0 text-2xl font-bold">
          {position.toString().padStart(2, "0")}
        </span>
        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          <span className="truncate font-medium">{track.filename}</span>
          <div className="flex flex-wrap gap-1.5">
            <Chip>{formatSeconds(track.duration_seconds)}</Chip>
            {track.bpm != null && <Chip>{Math.round(track.bpm)} BPM</Chip>}
            {track.key_camelot && <Chip>{track.key_camelot}</Chip>}
          </div>
        </div>
      </div>
      {track.analysis_status === "error" && (
        <p className="text-sm text-red-300">{track.analysis_error}</p>
      )}
      <audio controls preload="none" src={trackAudioUrl(track.id)} className="h-10 w-full" />
    </li>
  );
}

import type { DragEvent } from "react";

import { trackAudioUrl, type Track } from "@/lib/api";
import { formatSeconds } from "@/lib/format";

function Chip({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded-full bg-white/8 px-2.5 py-0.5 font-mono text-xs text-neutral-300">
      {children}
    </span>
  );
}

export type Reorder = {
  canUp: boolean;
  canDown: boolean;
  onMove: (delta: -1 | 1) => void;
  onDragStart: () => void;
  onDragEnter: () => void;
  onDrop: () => void;
  onDragEnd: () => void;
  dragging: boolean;
  dropTarget: boolean;
};

const arrowClasses =
  "flex h-7 w-7 items-center justify-center rounded-full border border-white/15 text-neutral-300 transition hover:border-white/35 hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-30";

export default function TrackCard({
  track,
  position,
  reorder,
}: {
  track: Track;
  position: number;
  reorder?: Reorder;
}) {
  const name = track.filename.replace(/\.[^.]+$/, "");
  const allowDrop = (e: DragEvent) => e.preventDefault();

  return (
    <li
      className={`glass animate-fade-up flex flex-col gap-3 rounded-2xl p-4 transition ${
        reorder?.dragging ? "opacity-40" : ""
      } ${reorder?.dropTarget ? "ring-neon-cyan ring-2" : ""}`}
      onDragEnter={reorder?.onDragEnter}
      onDragOver={reorder ? allowDrop : undefined}
      onDrop={
        reorder
          ? (e) => {
              e.preventDefault();
              reorder.onDrop();
            }
          : undefined
      }
    >
      <div className="flex items-center gap-4">
        {reorder && (
          <span
            draggable
            onDragStart={(e) => {
              e.dataTransfer.effectAllowed = "move";
              e.dataTransfer.setData("text/plain", track.id);
              reorder.onDragStart();
            }}
            onDragEnd={reorder.onDragEnd}
            aria-hidden
            title="Drag to reorder"
            className="cursor-grab text-lg leading-none text-neutral-500 select-none hover:text-neutral-300 active:cursor-grabbing"
          >
            ⠿
          </span>
        )}
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
        {reorder && (
          <div className="flex shrink-0 flex-col gap-1">
            <button
              type="button"
              className={arrowClasses}
              disabled={!reorder.canUp}
              onClick={() => reorder.onMove(-1)}
              aria-label={`Move ${name} earlier`}
            >
              ↑
            </button>
            <button
              type="button"
              className={arrowClasses}
              disabled={!reorder.canDown}
              onClick={() => reorder.onMove(1)}
              aria-label={`Move ${name} later`}
            >
              ↓
            </button>
          </div>
        )}
      </div>
      {track.analysis_status === "error" && (
        <p className="text-sm text-red-300">{track.analysis_error}</p>
      )}
      <audio controls preload="none" src={trackAudioUrl(track.id)} className="h-10 w-full" />
    </li>
  );
}

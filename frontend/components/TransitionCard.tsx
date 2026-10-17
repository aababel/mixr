import type { ReactNode } from "react";

import type { Track } from "@/lib/api";
import { formatSeconds } from "@/lib/format";
import { trackColor, type Blend } from "@/lib/timeline";

type Props = {
  blend: Blend;
  from?: Track;
  to?: Track;
  onPlay: () => void;
};

const GAIN_EPSILON_DB = 0.05;

function gainText(name: string, db: number | undefined) {
  if (db === undefined || Math.abs(db) < GAIN_EPSILON_DB) return null;
  return `${name} turned ${db < 0 ? "down" : "up"} ${Math.abs(db).toFixed(1)} dB`;
}

function Pill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "neon" | "warn" }) {
  const tones = {
    neutral: "border-white/10 bg-white/5 text-neutral-200",
    neon: "border-neon-pink/40 bg-neon-pink/10 text-pink-100",
    warn: "border-amber-400/30 bg-amber-400/10 text-amber-200",
  };
  return <span className={`rounded-full border px-2.5 py-0.5 text-xs ${tones[tone]}`}>{children}</span>;
}

export default function TransitionCard({ blend, from, to, onPlay }: Props) {
  const t = blend.transition;
  const fromNumber = blend.index + 1;
  const toNumber = blend.index + 2;
  const beatsAligned = !t.safety_adjustments.some((s) => s.startsWith("beat alignment skipped"));
  const gains = [
    gainText(`Song ${fromNumber}`, t.applied_gain_db.outgoing),
    gainText(`Song ${toNumber}`, t.applied_gain_db.incoming),
  ].filter(Boolean);

  return (
    <li className="flex flex-col gap-3 rounded-2xl bg-white/[0.03] p-4">
      <div className="flex items-center gap-3">
        <span className="font-display text-lg font-bold" style={{ color: trackColor(blend.index) }}>
          {fromNumber.toString().padStart(2, "0")}
        </span>
        <span className="text-neon font-bold">→</span>
        <span className="font-display text-lg font-bold" style={{ color: trackColor(blend.index + 1) }}>
          {toNumber.toString().padStart(2, "0")}
        </span>
        <span className="min-w-0 flex-1 truncate text-sm text-neutral-400">
          {from?.filename} → {to?.filename}
        </span>
        <button
          type="button"
          onClick={onPlay}
          className="shrink-0 rounded-full border border-white/15 px-3 py-1 font-mono text-xs text-neutral-200 transition hover:border-neon-pink/60 hover:text-white"
        >
          ▶ {formatSeconds(blend.start)}
        </button>
      </div>

      <div className="flex flex-wrap gap-2">
        <Pill tone="neon">{t.actual_overlap_seconds.toFixed(1)}s blend</Pill>
        {beatsAligned && <Pill>Beats aligned</Pill>}
        {from?.bpm != null && to?.bpm != null && (
          <Pill>
            {Math.round(from.bpm)} → {Math.round(to.bpm)} BPM
          </Pill>
        )}
        {t.tempo_change != null && <Pill tone="neon">Tempo glide ×{t.tempo_change.toFixed(2)}</Pill>}
        {t.applied_eq && <Pill tone="neon">{t.applied_eq.replaceAll("_", " ")}</Pill>}
        {gains.length > 0 && <Pill>Volume matched</Pill>}
        {t.overlap_changed_for_safety && <Pill tone="warn">Shortened to fit</Pill>}
      </div>

      {gains.length > 0 && <p className="text-xs text-neutral-400">{gains.join(" · ")}</p>}

      {t.safety_adjustments.length > 0 && (
        <details className="text-xs text-neutral-500">
          <summary className="cursor-pointer select-none hover:text-neutral-300">Engine notes</summary>
          <ul className="mt-1 list-disc pl-5">
            {t.safety_adjustments.map((s) => (
              <li key={s}>{s}</li>
            ))}
          </ul>
        </details>
      )}
    </li>
  );
}

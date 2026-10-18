import type { MixOptions } from "@/lib/api";
import { formatSeconds } from "@/lib/format";

const SECONDS_PER_SONG = { Short: 60, Medium: 90, Long: 150 } as const;
const MIN_SET_SECONDS = 120;

/** Default DJ set length for a number of songs (the "Medium" option). */
export const defaultSetSeconds = (songs: number) =>
  Math.max(MIN_SET_SECONDS, songs * SECONDS_PER_SONG.Medium);

type Props = {
  songs: number;
  fullSeconds: number;
  options: MixOptions;
  onChange: (options: MixOptions) => void;
};

export default function ModePicker({ songs, fullSeconds, options, onChange }: Props) {
  const lengths = Object.entries(SECONDS_PER_SONG).map(([label, perSong]) => ({
    label,
    seconds: Math.max(MIN_SET_SECONDS, songs * perSong),
  }));

  const modes = [
    {
      mode: "full" as const,
      title: "Full songs",
      detail: `Every song, trimmed only to hand off cleanly · up to ${formatSeconds(fullSeconds)}`,
    },
    {
      mode: "set" as const,
      title: "DJ set",
      detail: "Best part of each song, blended tight",
    },
  ];

  return (
    <section className="glass flex flex-col gap-4 rounded-2xl p-4">
      <h3 className="text-xs tracking-widest text-neutral-400 uppercase">Mix style</h3>
      <div role="radiogroup" aria-label="Mix style" className="grid gap-2 sm:grid-cols-2">
        {modes.map((m) => {
          const selected = options.mode === m.mode;
          return (
            <button
              key={m.mode}
              type="button"
              role="radio"
              aria-checked={selected}
              onClick={() =>
                onChange({
                  mode: m.mode,
                  targetSeconds: m.mode === "set" ? (options.targetSeconds ?? defaultSetSeconds(songs)) : null,
                })
              }
              className={`flex flex-col gap-1 rounded-xl border px-4 py-3 text-left transition ${
                selected
                  ? "glow border-neon-pink/60 bg-neon-pink/10"
                  : "border-white/10 bg-white/[0.03] hover:border-white/25"
              }`}
            >
              <span className="font-display font-semibold">{m.title}</span>
              <span className="text-xs text-neutral-400">{m.detail}</span>
            </button>
          );
        })}
      </div>

      {options.mode === "set" && (
        <div className="flex flex-col gap-2">
          <span className="text-xs text-neutral-400">Set length</span>
          <div role="radiogroup" aria-label="Set length" className="flex flex-wrap gap-2">
            {lengths.map((l) => {
              const selected = options.targetSeconds === l.seconds;
              return (
                <button
                  key={l.label}
                  type="button"
                  role="radio"
                  aria-checked={selected}
                  onClick={() => onChange({ mode: "set", targetSeconds: l.seconds })}
                  className={`rounded-full border px-3 py-1 text-sm transition ${
                    selected
                      ? "border-neon-pink/60 bg-neon-pink/15 text-white"
                      : "border-white/10 text-neutral-300 hover:border-white/25"
                  }`}
                >
                  {l.label} · {formatSeconds(l.seconds)}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}

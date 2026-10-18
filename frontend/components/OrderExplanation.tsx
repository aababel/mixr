import type { OrderExplanation as Explanation, Track } from "@/lib/api";

const stripExtension = (filename: string) => filename.replace(/\.[^.]+$/, "");

export default function OrderExplanation({
  explanation,
  tracks,
  onReset,
}: {
  explanation: Explanation;
  tracks: Track[];
  /** Set when the user rearranged the songs: puts back Mixr's pick. */
  onReset?: () => void;
}) {
  const nameOf = new Map(tracks.map((t) => [t.id, stripExtension(t.filename)]));

  return (
    <section className="glass animate-fade-up flex flex-col gap-3 rounded-2xl p-4">
      <div className="flex items-center justify-between gap-3">
        <h2 className="font-display text-neon text-sm font-semibold tracking-wide uppercase">
          {onReset ? "Your order" : "Why this order"}
        </h2>
        {onReset && (
          <button
            type="button"
            onClick={onReset}
            className="rounded-full border border-white/15 px-3 py-1 text-xs text-neutral-300 transition hover:border-white/35 hover:bg-white/5"
          >
            Use Mixr&apos;s order
          </button>
        )}
      </div>
      <p className="font-medium">{explanation.summary}</p>
      {explanation.transitions.length > 0 && (
        <ul className="flex flex-col gap-2 text-sm text-neutral-300">
          {explanation.transitions.map((t) => (
            <li key={`${t.from_id}-${t.to_id}`}>
              <span className="font-medium text-neutral-100">
                {nameOf.get(t.from_id) ?? t.from_id} → {nameOf.get(t.to_id) ?? t.to_id}
              </span>
              <span className="text-neutral-400">
                : {t.reasons.join("; ")} (score {Math.round(t.score * 100)}%)
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

import type { OrderExplanation as Explanation, Track } from "@/lib/api";

const stripExtension = (filename: string) => filename.replace(/\.[^.]+$/, "");

export default function OrderExplanation({
  explanation,
  tracks,
}: {
  explanation: Explanation;
  tracks: Track[];
}) {
  const nameOf = new Map(tracks.map((t) => [t.id, stripExtension(t.filename)]));

  return (
    <section className="glass animate-fade-up flex flex-col gap-3 rounded-2xl p-4">
      <h2 className="font-display text-neon text-sm font-semibold tracking-wide uppercase">
        Why this order
      </h2>
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

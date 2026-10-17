import { trackColor } from "@/lib/timeline";

/** "Why this order": summary line, then one "A → B: reasons (score N%)" line per transition. */
export default function OrderExplanation({ lines }: { lines: string[] }) {
  const [summary, ...transitions] = lines;
  if (!summary) return null;

  return (
    <section className="glass animate-fade-up flex flex-col gap-3 rounded-2xl p-5">
      <h3 className="text-xs tracking-widest text-neutral-400 uppercase">Why this order</h3>
      <p className="font-medium text-neutral-100">{summary}</p>
      {transitions.length > 0 && (
        <ol className="flex flex-col gap-2 text-sm">
          {transitions.map((line, i) => {
            const split = line.indexOf(": ");
            const pair = split === -1 ? null : line.slice(0, split);
            const reasons = split === -1 ? line : line.slice(split + 2);
            return (
              <li key={line} className="flex gap-3">
                <span
                  className="mt-1.5 size-2 shrink-0 rounded-full"
                  style={{ background: trackColor(i + 1) }}
                  aria-hidden
                />
                <span className="text-neutral-400">
                  {pair && <span className="text-neutral-200">{pair}: </span>}
                  {reasons}
                </span>
              </li>
            );
          })}
        </ol>
      )}
    </section>
  );
}

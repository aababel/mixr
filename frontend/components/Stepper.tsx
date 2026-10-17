export default function Stepper({ steps, current }: { steps: string[]; current: number }) {
  return (
    <ol className="flex items-center gap-2 text-xs sm:text-sm">
      {steps.map((label, i) => {
        const done = i < current;
        const active = i === current;
        return (
          <li
            key={label}
            className={`flex items-center gap-2 ${i < steps.length - 1 ? "flex-1" : ""}`}
          >
            <span
              className={`flex size-7 shrink-0 items-center justify-center rounded-full text-xs font-semibold transition ${
                active
                  ? "bg-neon text-white glow"
                  : done
                    ? "bg-white/15 text-white"
                    : "border border-white/15 text-neutral-500"
              }`}
            >
              {done ? "✓" : i + 1}
            </span>
            <span
              className={`hidden sm:inline ${active ? "text-white" : done ? "text-neutral-300" : "text-neutral-500"}`}
            >
              {label}
            </span>
            {i < steps.length - 1 && (
              <span className={`h-px flex-1 ${done ? "bg-neon" : "bg-white/10"}`} />
            )}
          </li>
        );
      })}
    </ol>
  );
}

const DELAYS = [0, 0.35, 0.15, 0.5, 0.25, 0.6, 0.1, 0.45];

export default function Equalizer({
  bars = 5,
  className = "h-6",
  barClassName = "w-1",
}: {
  bars?: number;
  className?: string;
  barClassName?: string;
}) {
  return (
    <span aria-hidden className={`inline-flex items-end gap-[3px] ${className}`}>
      {Array.from({ length: bars }, (_, i) => (
        <span
          key={i}
          className={`animate-eq h-full origin-bottom rounded-full bg-neon ${barClassName}`}
          style={{ animationDelay: `${-DELAYS[i % DELAYS.length]}s` }}
        />
      ))}
    </span>
  );
}

export default function Vinyl({
  className = "size-28",
  spinning = true,
}: {
  className?: string;
  spinning?: boolean;
}) {
  return (
    <div
      aria-hidden
      className={`relative rounded-full shadow-[0_0_40px_-8px_var(--color-neon-pink)] ${
        spinning ? "animate-spin-slow" : ""
      } ${className}`}
      style={{
        background: "repeating-radial-gradient(circle at center, #16141f 0 2px, #0b0a10 2px 4px)",
      }}
    >
      <div className="absolute inset-[34%] rounded-full bg-neon" />
      <div className="absolute inset-[47%] rounded-full bg-night" />
      <div className="absolute inset-0 rounded-full bg-[conic-gradient(from_200deg,transparent_0_40%,rgb(255_255_255/0.08)_50%,transparent_60%)]" />
    </div>
  );
}

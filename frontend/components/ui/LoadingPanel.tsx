import Equalizer from "./Equalizer";
import Vinyl from "./Vinyl";

export default function LoadingPanel({ title, detail }: { title: string; detail: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="glass animate-fade-up flex flex-col items-center gap-6 rounded-3xl px-6 py-14 text-center"
    >
      <Vinyl />
      <div className="flex flex-col items-center gap-2">
        <p className="font-display text-xl">{title}</p>
        <p className="text-sm text-neutral-400">{detail}</p>
      </div>
      <Equalizer bars={8} className="h-8" barClassName="w-1.5" />
    </div>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";

import { mixAudioUrl, type MixResult as Mix, type Track } from "@/lib/api";
import { formatSeconds } from "@/lib/format";
import { BLEND_PREROLL_SECONDS, activeIndex, mixLayout, trackColor } from "@/lib/timeline";

import MixTimeline from "./MixTimeline";
import TransitionCard from "./TransitionCard";
import { buttonClasses } from "./ui/Button";
import Equalizer from "./ui/Equalizer";
import Vinyl from "./ui/Vinyl";

export default function MixResult({ mix, tracks }: { mix: Mix; tracks: Track[] }) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const { manifest } = mix;
  const byId = new Map(tracks.map((t) => [t.id, t]));
  const name = (id: string) => byId.get(id)?.filename ?? id;
  const { total, segments, blends } = mixLayout(manifest);
  const nowPlaying = activeIndex(segments, currentTime);

  useEffect(() => {
    if (!playing) return;
    let frame = 0;
    const tick = () => {
      setCurrentTime(audioRef.current?.currentTime ?? 0);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing]);

  function seek(seconds: number) {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = seconds;
    setCurrentTime(seconds);
    audio.play();
  }

  return (
    <div className="animate-fade-up flex flex-col gap-6">
      <section className="glass flex flex-col gap-6 rounded-3xl p-6">
        <div className="flex items-center gap-5">
          <Vinyl className="size-20 shrink-0" spinning={playing} />
          <div className="flex min-w-0 flex-col gap-1">
            <span className="text-xs tracking-widest text-neutral-400 uppercase">Your mix</span>
            <h2 className="font-display text-neon text-3xl font-bold">Ready to play</h2>
            <span className="text-sm text-neutral-400">
              {manifest.ordered_track_ids.length} songs ·{" "}
              {formatSeconds(manifest.output_duration_seconds)} ·{" "}
              {manifest.calibrated ? "tuned to your taste" : "default transitions"}
            </span>
          </div>
          {playing && <Equalizer bars={5} className="ml-auto hidden h-8 sm:inline-flex" />}
        </div>

        <MixTimeline
          total={total}
          segments={segments}
          blends={blends}
          tracks={byId}
          currentTime={currentTime}
          onSeek={seek}
        />

        <audio
          ref={audioRef}
          controls
          src={mixAudioUrl(mix)}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onSeeked={(e) => setCurrentTime(e.currentTarget.currentTime)}
          className="w-full"
        />

        <a href={mixAudioUrl(mix)} className={buttonClasses("primary")}>
          Download mix (WAV)
        </a>
      </section>

      <section className="glass flex flex-col gap-3 rounded-3xl p-6">
        <h3 className="text-xs tracking-widest text-neutral-400 uppercase">Tracklist</h3>
        <ol className="flex flex-col">
          {segments.map((s) => {
            const active = s.index === nowPlaying && currentTime > 0;
            return (
              <li key={s.id}>
                <button
                  onClick={() => seek(s.start)}
                  className={`group flex w-full items-center gap-4 rounded-xl px-2 py-2.5 text-left transition hover:bg-white/5 ${
                    active ? "bg-white/[0.06]" : ""
                  }`}
                >
                  <span
                    className="font-display flex w-8 shrink-0 justify-center text-lg font-bold"
                    style={{ color: trackColor(s.index) }}
                  >
                    {active && playing ? (
                      <Equalizer bars={3} className="h-4" />
                    ) : (
                      (s.index + 1).toString().padStart(2, "0")
                    )}
                  </span>
                  <span className={`flex-1 truncate ${active ? "text-white" : ""}`}>{name(s.id)}</span>
                  <span className="font-mono text-sm text-neutral-400 group-hover:text-white">
                    {formatSeconds(s.start)}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      </section>

      {blends.length > 0 && (
        <section className="glass flex flex-col gap-3 rounded-3xl p-6">
          <h3 className="text-xs tracking-widest text-neutral-400 uppercase">How Mixr blended them</h3>
          <ul className="flex flex-col gap-3">
            {blends.map((b) => (
              <TransitionCard
                key={b.index}
                blend={b}
                from={byId.get(b.transition.from_id)}
                to={byId.get(b.transition.to_id)}
                onPlay={() => seek(Math.max(0, b.start - BLEND_PREROLL_SECONDS))}
              />
            ))}
          </ul>
        </section>
      )}

      {manifest.limitations.length > 0 && (
        <section className="flex flex-col gap-1 px-2 text-sm text-neutral-500">
          <h3 className="text-xs tracking-widest uppercase">Limitations</h3>
          <ul className="list-disc pl-5">
            {manifest.limitations.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}

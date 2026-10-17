"use client";

import { useState, type Ref } from "react";

import Equalizer from "./ui/Equalizer";

type Props = {
  min: number;
  max: number;
  onFiles: (files: FileList | null) => void;
  ref?: Ref<HTMLInputElement>;
};

export default function Dropzone({ min, max, onFiles, ref }: Props) {
  const [dragging, setDragging] = useState(false);

  return (
    <label
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        onFiles(e.dataTransfer.files);
      }}
      className={`group relative flex cursor-pointer flex-col items-center gap-4 overflow-hidden rounded-3xl border-2 border-dashed px-6 py-16 text-center transition ${
        dragging
          ? "border-neon-pink bg-neon-pink/10 glow"
          : "glass border-white/15 hover:border-neon-violet/60"
      }`}
    >
      <span className="flex size-16 items-center justify-center rounded-full bg-white/5 transition group-hover:scale-105">
        <Equalizer bars={5} className="h-7" barClassName="w-1.5" />
      </span>
      <span className="font-display text-xl">
        {dragging ? "Drop it" : `Drop ${min}–${max} songs here`}
      </span>
      <span className="text-sm text-neutral-400">or click to choose MP3 or WAV files</span>
      <input
        ref={ref}
        type="file"
        multiple
        accept=".mp3,.wav,audio/mpeg,audio/wav"
        className="sr-only"
        onChange={(e) => onFiles(e.target.files)}
      />
    </label>
  );
}

import Link from "next/link";

import SiteHeader from "@/components/SiteHeader";
import { buttonClasses } from "@/components/ui/Button";
import Equalizer from "@/components/ui/Equalizer";
import Vinyl from "@/components/ui/Vinyl";

const STEPS = [
  {
    title: "Drop in your songs",
    body: "Upload 2–5 MP3 or WAV files. Nothing to sign into.",
  },
  {
    title: "Mixr reads the music",
    body: "Tempo, beats, key and energy for every track, then the order that flows best.",
  },
  {
    title: "Get one seamless mix",
    body: "Every song plays in full, blended at the seams. Play it here or download it.",
  },
];

export default function Home() {
  return (
    <>
      <SiteHeader>
        <Link href="/mix" className={buttonClasses("ghost", "px-5 py-2 text-sm")}>
          Open app
        </Link>
      </SiteHeader>

      <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-24 px-6 pb-16">
        <section className="grid items-center gap-12 pt-8 md:grid-cols-[1.2fr_1fr] md:pt-16">
          <div className="animate-fade-up flex flex-col gap-6">
            <span className="glass w-fit rounded-full px-3 py-1 text-xs tracking-wide text-neutral-300">
              Built at DubHacks 2026
            </span>
            <h1 className="font-display text-5xl leading-[1.05] font-bold tracking-tight sm:text-6xl">
              Your playlist.
              <br />
              <span className="text-neon">DJ&apos;d by AI.</span>
            </h1>
            <p className="max-w-md text-lg text-neutral-400">
              Hand Mixr a few songs. It figures out the best order and blends them into one
              continuous mix, like a DJ would.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <Link href="/mix" className={buttonClasses("primary", "px-8 py-4 text-lg")}>
                Start mixing
              </Link>
              <a href="#how" className={buttonClasses("ghost", "px-6 py-4")}>
                How it works
              </a>
            </div>
          </div>

          <div
            className="animate-fade-up relative mx-auto flex items-center justify-center"
            style={{ animationDelay: "120ms" }}
          >
            <Vinyl className="size-64 sm:size-80" />
            <div className="glass absolute -bottom-4 left-1/2 flex -translate-x-1/2 items-center gap-3 rounded-full px-4 py-2">
              <Equalizer bars={6} className="h-5" barClassName="w-1" />
              <span className="text-sm whitespace-nowrap text-neutral-300">Now mixing</span>
            </div>
          </div>
        </section>

        <section id="how" className="flex scroll-mt-8 flex-col gap-8">
          <h2 className="font-display text-2xl font-semibold">How it works</h2>
          <ol className="grid gap-4 md:grid-cols-3">
            {STEPS.map((step, i) => (
              <li key={step.title} className="glass flex flex-col gap-3 rounded-3xl p-6">
                <span className="font-display text-neon text-3xl font-bold">0{i + 1}</span>
                <h3 className="text-lg font-semibold">{step.title}</h3>
                <p className="text-sm text-neutral-400">{step.body}</p>
              </li>
            ))}
          </ol>
        </section>
      </main>

      <footer className="mx-auto w-full max-w-5xl px-6 py-8 text-sm text-neutral-500">
        Made by Shaan, Aarav and Sahil.
      </footer>
    </>
  );
}

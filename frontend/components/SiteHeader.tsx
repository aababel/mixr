import Link from "next/link";

import Equalizer from "./ui/Equalizer";

export default function SiteHeader({ children }: { children?: React.ReactNode }) {
  return (
    <header className="mx-auto flex w-full max-w-5xl items-center justify-between px-6 py-6">
      <Link href="/" className="group flex items-center gap-2.5">
        <Equalizer bars={4} className="h-5" barClassName="w-[3px]" />
        <span className="font-display text-xl font-semibold tracking-tight">mixr</span>
      </Link>
      {children}
    </header>
  );
}

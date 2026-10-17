import type { Metadata } from "next";
import { Geist, Geist_Mono, Unbounded } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

const unbounded = Unbounded({
  variable: "--font-unbounded",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Mixr",
  description: "Your playlist. DJ'd by AI.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${unbounded.variable} h-full antialiased`}
    >
      <body className="relative min-h-full flex flex-col overflow-x-hidden">
        <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
          <div className="animate-drift absolute -left-1/4 -top-1/3 h-[70vh] w-[70vw] rounded-full bg-neon-pink/20 blur-[120px]" />
          <div
            className="animate-drift absolute -bottom-1/3 -right-1/4 h-[70vh] w-[70vw] rounded-full bg-neon-cyan/15 blur-[120px]"
            style={{ animationDelay: "-9s" }}
          />
        </div>
        {children}
      </body>
    </html>
  );
}

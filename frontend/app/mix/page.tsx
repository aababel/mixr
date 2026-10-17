import type { Metadata } from "next";

import Mixr from "@/components/Mixr";
import SiteHeader from "@/components/SiteHeader";

export const metadata: Metadata = {
  title: "Mixr · New mix",
};

export default function MixPage() {
  return (
    <>
      <SiteHeader />
      <main className="flex flex-1 flex-col">
        <Mixr />
      </main>
    </>
  );
}

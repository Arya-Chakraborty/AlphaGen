import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "molecule-gen",
  description: "Generate molecules that match Mordred 2D descriptor ranges.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      {/* Extensions inject attributes (e.g. style="cursor: auto") onto body before
          React hydrates. Shallow, so real mismatches inside the app still warn. */}
      <body suppressHydrationWarning>{children}</body>
    </html>
  );
}

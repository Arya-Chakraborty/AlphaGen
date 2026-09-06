import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "molecule-gen",
  description: "Generate molecules that match Mordred 2D descriptor ranges.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

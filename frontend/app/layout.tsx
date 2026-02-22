import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Assistant — Multi-Source AI",
  description: "Multi-source agentic RAG application powered by Gemini and Qdrant",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}

import type React from "react"
import type { Metadata } from "next"
import "./globals.css"
import AuthSessionProvider from "@/components/auth-session-provider"

export const metadata: Metadata = {
  title: "PropertyGlow - AI-Powered Real Estate Marketing",
  description:
    "Transform your property listings with AI-enhanced images, compelling descriptions, and professional marketing assets.",
  generator: "v0.dev",
  viewport: "width=device-width, initial-scale=1",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className="antialiased">
        <AuthSessionProvider>{children}</AuthSessionProvider>
      </body>
    </html>
  )
}

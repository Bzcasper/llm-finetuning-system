import type React from "react"
import type { Metadata } from "next"

export const metadata: Metadata = {
  title: "PropertyGlow Dashboard",
  description: "Enhance your real estate photos with AI",
}

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return <div className="min-h-screen bg-gray-50">{children}</div>
}

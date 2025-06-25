import type React from "react"
import { Card, CardContent } from "@/components/ui/card"

interface FeatureCardProps {
  icon: React.ReactNode
  title: string
  description: string
}

export default function FeatureCard({ icon, title, description }: FeatureCardProps) {
  return (
    <Card className="h-full border-0 shadow-sm hover:shadow-md transition-shadow duration-200">
      <CardContent className="p-6 sm:p-8 text-center space-y-4">
        <div className="flex justify-center">{icon}</div>
        <h3 className="text-lg sm:text-xl font-semibold text-gray-900">{title}</h3>
        <p className="text-sm sm:text-base text-gray-600 leading-relaxed">{description}</p>
      </CardContent>
    </Card>
  )
}

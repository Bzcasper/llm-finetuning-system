"use client"

import type React from "react"

import { useState, useRef } from "react"
import Image from "next/image"

interface BeforeAfterSliderProps {
  beforeImage: string
  afterImage: string
  className?: string
}

export default function BeforeAfterSlider({ beforeImage, afterImage, className = "" }: BeforeAfterSliderProps) {
  const [sliderPosition, setSliderPosition] = useState(50)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!containerRef.current) return

    const rect = containerRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width))
    const percentage = Math.max(0, Math.min((x / rect.width) * 100, 100))

    setSliderPosition(percentage)
  }

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!containerRef.current) return

    const rect = containerRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(e.touches[0].clientX - rect.left, rect.width))
    const percentage = Math.max(0, Math.min((x / rect.width) * 100, 100))

    setSliderPosition(percentage)
  }

  return (
    <div
      ref={containerRef}
      className={`relative w-full max-w-2xl aspect-[4/3] cursor-ew-resize overflow-hidden ${className}`}
      onMouseMove={handleMouseMove}
      onTouchMove={handleTouchMove}
    >
      {/* Before Image (Full width) */}
      <div className="absolute inset-0">
        <Image src={beforeImage || "/placeholder.svg"} alt="Before enhancement" fill className="object-cover" />
      </div>

      {/* After Image (Clipped) */}
      <div className="absolute inset-0 overflow-hidden" style={{ width: `${sliderPosition}%` }}>
        <Image src={afterImage || "/placeholder.svg"} alt="After enhancement" fill className="object-cover" />
      </div>

      {/* Slider Control */}
      <div
        className="absolute top-0 bottom-0 w-1 bg-white shadow-md cursor-ew-resize"
        style={{ left: `${sliderPosition}%`, transform: "translateX(-50%)" }}
      >
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-6 h-6 bg-white rounded-full shadow-md flex items-center justify-center">
          <div className="flex flex-col gap-1">
            <div className="w-1 h-3 bg-gray-400 rounded-full"></div>
            <div className="w-1 h-3 bg-gray-400 rounded-full"></div>
          </div>
        </div>
      </div>

      {/* Labels */}
      <div className="absolute bottom-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">Before</div>
      <div className="absolute bottom-2 right-2 bg-black/50 text-white text-xs px-2 py-1 rounded">After</div>
    </div>
  )
}

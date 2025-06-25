"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Loader2 } from "lucide-react"
import Link from "next/link"

export default function FalExamplePage() {
  const [prompt, setPrompt] = useState<string>("a beautiful house with a garden")
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const generateImage = async () => {
    if (!prompt) {
      setError("Please enter a prompt")
      return
    }

    setIsGenerating(true)
    setError(null)
    setImageUrl(null)

    try {
      const response = await fetch("/api/generate-image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to generate image")
      }

      if (!data.imageUrl) {
        throw new Error("No image URL was returned")
      }

      setImageUrl(data.imageUrl)
    } catch (err) {
      console.error("Error generating image:", err)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <div className="container max-w-4xl py-8">
      <div className="mb-6">
        <Link href="/dashboard">
          <Button variant="outline" size="sm">
            Back to Dashboard
          </Button>
        </Link>
      </div>

      <h1 className="text-2xl font-bold mb-4">Fal.ai Example</h1>
      <p className="text-gray-500 mb-6">Generate an image using fal.ai's Flux model</p>

      <div className="space-y-6">
        <div className="border p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-4">Enter a Prompt</h2>
          <div className="space-y-4">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="w-full p-2 border rounded-md"
              rows={3}
              placeholder="Enter a description of the image you want to generate..."
            />
            <Button
              onClick={generateImage}
              disabled={isGenerating || !prompt}
              className="bg-rose-500 hover:bg-rose-600"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Generating...
                </>
              ) : (
                "Generate Image"
              )}
            </Button>
          </div>
        </div>

        {error && (
          <div className="p-3 bg-red-50 text-red-700 rounded-md">
            <p className="font-medium">{error}</p>
          </div>
        )}

        {imageUrl && (
          <div className="border p-4 rounded-lg">
            <h2 className="text-lg font-medium mb-2">Generated Image</h2>
            <div className="aspect-square bg-gray-100 rounded-lg overflow-hidden">
              <img src={imageUrl || "/placeholder.svg"} alt="Generated" className="w-full h-full object-contain" />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

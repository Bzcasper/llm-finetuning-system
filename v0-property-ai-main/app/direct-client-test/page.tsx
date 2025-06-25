"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Loader2 } from "lucide-react"

export default function DirectClientTestPage() {
  const [result, setResult] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [errorDetails, setErrorDetails] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])

  const addLog = (message: string) => {
    setLogs((prev) => [...prev, `[${new Date().toISOString()}] ${message}`])
  }

  const generateImage = async () => {
    setIsProcessing(true)
    setError(null)
    setErrorDetails(null)
    setResult(null)
    addLog("Starting image generation")

    try {
      // Use our server-side API endpoint instead of direct client-side calls
      addLog("Sending request to server-side API")
      const response = await fetch("/api/generate-image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: "a beautiful house with a garden",
        }),
      })

      addLog(`Received response: ${response.status} ${response.statusText}`)
      const data = await response.json()

      if (!response.ok) {
        addLog(`Error response: ${JSON.stringify(data)}`)
        setErrorDetails(data.details || JSON.stringify(data))
        throw new Error(data.error || "Failed to generate image")
      }

      if (!data.imageUrl) {
        addLog("No image URL in response")
        throw new Error("No image URL was returned")
      }

      addLog(`Successfully received image URL`)
      setResult(data.imageUrl)
    } catch (err) {
      console.error("Error generating image:", err)
      addLog(`Error: ${err instanceof Error ? err.message : "Unknown error"}`)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="container max-w-4xl py-8">
      <h1 className="text-2xl font-bold mb-4">Direct Client-Side Fal AI Test</h1>
      <p className="text-gray-500 mb-6">This test uses a server-side API endpoint to generate an image with Fal AI.</p>

      <div className="space-y-6">
        <div className="border p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-2">Generate an Image</h2>
          <p className="text-sm text-gray-500 mb-4">
            This will generate an image of "a beautiful house with a garden" using the flux/dev model.
          </p>
          <Button onClick={generateImage} disabled={isProcessing} className="bg-rose-500 hover:bg-rose-600">
            {isProcessing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Generating...
              </>
            ) : (
              "Generate Image"
            )}
          </Button>

          {error && (
            <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md">
              <p className="font-medium">{error}</p>
              {errorDetails && (
                <div className="mt-2 p-2 bg-red-100 rounded overflow-auto text-xs">
                  <pre>{errorDetails}</pre>
                </div>
              )}
            </div>
          )}
        </div>

        {result && (
          <div className="border p-4 rounded-lg">
            <h2 className="text-lg font-medium mb-2">Result</h2>
            <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
              <img src={result || "/placeholder.svg"} alt="Generated" className="w-full h-full object-contain" />
            </div>
          </div>
        )}

        <div className="border p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-2">Logs</h2>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-md font-mono text-xs h-64 overflow-y-auto">
            {logs.length > 0 ? (
              logs.map((log, i) => <div key={i}>{log}</div>)
            ) : (
              <div className="text-gray-500">No logs yet</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

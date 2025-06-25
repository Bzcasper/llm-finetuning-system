"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Loader2 } from "lucide-react"

export default function DirectTestPage() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [enhancedUrl, setEnhancedUrl] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])

  const addLog = (message: string) => {
    setLogs((prev) => [...prev, `[${new Date().toISOString()}] ${message}`])
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null)
    setEnhancedUrl(null)
    setLogs([])

    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      setPreviewUrl(URL.createObjectURL(selectedFile))
      addLog(`Selected file: ${selectedFile.name} (${selectedFile.type}, ${(selectedFile.size / 1024).toFixed(2)}KB)`)
    }
  }

  const processImage = async () => {
    if (!file) {
      setError("Please select an image first")
      return
    }

    setIsProcessing(true)
    setError(null)
    addLog("Starting direct Fal API test")

    try {
      // Convert file to base64
      const bytes = await file.arrayBuffer()
      const buffer = Buffer.from(bytes)
      const base64Image = `data:${file.type};base64,${buffer.toString("base64")}`

      addLog("Image converted to base64")

      // Send to our direct test API endpoint
      addLog("Sending request to direct test API")
      const response = await fetch("/api/direct-test", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: base64Image,
        }),
      })

      addLog(`Received response: ${response.status} ${response.statusText}`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to process image")
      }

      if (!data.enhancedImage) {
        throw new Error("No enhanced image was returned")
      }

      addLog("Successfully received enhanced image")
      setEnhancedUrl(data.enhancedImage)
    } catch (err) {
      console.error("Error processing image:", err)
      addLog(`Error: ${err instanceof Error ? err.message : "Unknown error"}`)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="container max-w-4xl py-8">
      <h1 className="text-2xl font-bold mb-4">Direct Fal API Test</h1>
      <p className="text-gray-500 mb-6">
        This test bypasses the AI SDK and uses the Fal API directly for troubleshooting.
      </p>

      <div className="space-y-6">
        <div className="border p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-2">1. Select an Image</h2>
          <input type="file" accept="image/*" onChange={handleFileChange} className="block w-full" />
        </div>

        {previewUrl && (
          <div className="border p-4 rounded-lg">
            <h2 className="text-lg font-medium mb-2">2. Preview</h2>
            <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
              <img src={previewUrl || "/placeholder.svg"} alt="Preview" className="w-full h-full object-contain" />
            </div>
          </div>
        )}

        <div className="border p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-2">3. Process</h2>
          <Button onClick={processImage} disabled={isProcessing || !file} className="bg-rose-500 hover:bg-rose-600">
            {isProcessing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              "Process Image"
            )}
          </Button>

          {error && <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md">{error}</div>}
        </div>

        {enhancedUrl && (
          <div className="border p-4 rounded-lg">
            <h2 className="text-lg font-medium mb-2">4. Result</h2>
            <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
              <img src={enhancedUrl || "/placeholder.svg"} alt="Enhanced" className="w-full h-full object-contain" />
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

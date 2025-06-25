"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Loader2 } from "lucide-react"

export default function SimpleTestPage() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [enhancedUrl, setEnhancedUrl] = useState<string | null>(null)
  const [requestId, setRequestId] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [errorDetails, setErrorDetails] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])

  const addLog = (message: string) => {
    setLogs((prev) => [...prev, `[${new Date().toISOString()}] ${message}`])
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null)
    setErrorDetails(null)
    setEnhancedUrl(null)
    setRequestId(null)
    setLogs([])

    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]

      // Check file size - recommend smaller files
      if (selectedFile.size > 1 * 1024 * 1024) {
        addLog(
          `Warning: File size is ${(selectedFile.size / 1024 / 1024).toFixed(2)}MB. Consider using a smaller image (< 1MB) for faster processing.`,
        )
      }

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
    setErrorDetails(null)
    addLog("Starting image processing")

    try {
      // Use a very simple enhancement option
      const enhancementOptions = {
        hdrEnhancement: true,
      }

      const formData = new FormData()
      formData.append("image", file)
      formData.append("options", JSON.stringify(enhancementOptions))

      addLog("Sending request to API")
      const response = await fetch("/api/enhance", {
        method: "POST",
        body: formData,
      })

      addLog(`Received response: ${response.status} ${response.statusText}`)
      const data = await response.json()

      if (!response.ok) {
        addLog(`Error response: ${JSON.stringify(data)}`)
        setErrorDetails(data.stack || data.details || JSON.stringify(data))
        throw new Error(data.error || "Failed to process image")
      }

      if (!data.enhancedImage) {
        addLog("No enhanced image in response")
        throw new Error("No enhanced image was returned")
      }

      addLog(`Successfully received enhanced image (Request ID: ${data.requestId})`)
      setEnhancedUrl(data.enhancedImage)
      setRequestId(data.requestId)
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
      <h1 className="text-2xl font-bold mb-4">Simple Fal AI Test</h1>
      <p className="text-gray-500 mb-4">This test uses the official @fal-ai/client library.</p>

      <div className="space-y-6">
        <div className="border p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-2">1. Select an Image</h2>
          <input type="file" accept="image/*" onChange={handleFileChange} className="block w-full" />
          <p className="text-xs text-gray-500 mt-2">
            Recommendation: Use a small image (under 1MB) for faster processing.
          </p>
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

        {enhancedUrl && (
          <div className="border p-4 rounded-lg">
            <h2 className="text-lg font-medium mb-2">4. Result</h2>
            {requestId && <p className="text-xs text-gray-500 mb-2">Request ID: {requestId}</p>}
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

"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Loader2 } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { falModels } from "@/lib/fal-models"
import Link from "next/link"

export default function ModelTestPage() {
  const [selectedModelId, setSelectedModelId] = useState<string>("clarity-upscaler")
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [enhancedUrl, setEnhancedUrl] = useState<string | null>(null)
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

    if (!selectedModelId) {
      setError("Please select a model")
      return
    }

    setIsProcessing(true)
    setError(null)
    setErrorDetails(null)
    setEnhancedUrl(null)
    addLog(`Starting image processing with model: ${selectedModelId}`)

    try {
      // Convert file to base64
      const bytes = await file.arrayBuffer()
      const buffer = Buffer.from(bytes)
      const base64Image = `data:${file.type};base64,${buffer.toString("base64")}`

      // Prepare form data
      const formData = new FormData()
      formData.append("image", file)
      formData.append("modelId", selectedModelId)

      // Send to our API
      addLog("Sending request to API")
      const response = await fetch("/api/enhance-with-model", {
        method: "POST",
        body: formData,
      })

      addLog(`Received response: ${response.status} ${response.statusText}`)
      const data = await response.json()

      if (!response.ok) {
        addLog(`Error response: ${JSON.stringify(data)}`)
        setErrorDetails(data.details || JSON.stringify(data))
        throw new Error(data.error || "Failed to process image")
      }

      if (!data.enhancedImage) {
        addLog("No enhanced image in response")
        throw new Error("No enhanced image was returned")
      }

      addLog(`Successfully received enhanced image`)
      setEnhancedUrl(data.enhancedImage)
    } catch (err) {
      console.error("Error processing image:", err)
      addLog(`Error: ${err instanceof Error ? err.message : "Unknown error"}`)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsProcessing(false)
    }
  }

  const selectedModel = falModels[selectedModelId]

  return (
    <div className="container max-w-4xl py-8">
      <div className="mb-6">
        <Link href="/dashboard">
          <Button variant="outline" size="sm">
            Back to Dashboard
          </Button>
        </Link>
      </div>

      <h1 className="text-2xl font-bold mb-4">Real Estate Image Enhancement Model Test</h1>
      <p className="text-gray-500 mb-6">
        Test different fal.ai models to find the best one for enhancing your real estate photos.
      </p>

      <div className="space-y-6">
        <div className="border p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-4">1. Select a Model</h2>
          <div className="space-y-4">
            <Select value={selectedModelId} onValueChange={setSelectedModelId}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a model" />
              </SelectTrigger>
              <SelectContent>
                {Object.values(falModels).map((model) => (
                  <SelectItem key={model.id} value={model.id}>
                    {model.name} ({model.type})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {selectedModel && (
              <div className="bg-gray-50 p-3 rounded-md">
                <p className="font-medium">{selectedModel.name}</p>
                <p className="text-sm text-gray-600">{selectedModel.description}</p>
                <p className="text-xs text-gray-500 mt-1">
                  Type: {selectedModel.type} | Commercial use:{" "}
                  {selectedModel.commercial ? (
                    <span className="text-green-600">Approved</span>
                  ) : (
                    <span className="text-yellow-600">Check licensing</span>
                  )}
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="border p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-2">2. Select an Image</h2>
          <input type="file" accept="image/*" onChange={handleFileChange} className="block w-full" />
          <p className="text-xs text-gray-500 mt-2">
            Recommendation: Use a small image (under 1MB) for faster processing.
          </p>
        </div>

        {previewUrl && (
          <div className="border p-4 rounded-lg">
            <h2 className="text-lg font-medium mb-2">3. Preview</h2>
            <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
              <img src={previewUrl || "/placeholder.svg"} alt="Preview" className="w-full h-full object-contain" />
            </div>
          </div>
        )}

        <div className="border p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-2">4. Process</h2>
          <Button
            onClick={processImage}
            disabled={isProcessing || !file || !selectedModelId}
            className="bg-rose-500 hover:bg-rose-600"
          >
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
            <h2 className="text-lg font-medium mb-2">5. Result</h2>
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

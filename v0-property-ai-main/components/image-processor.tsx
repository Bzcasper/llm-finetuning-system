"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Loader2, AlertCircle, RefreshCw } from "lucide-react"
import type { EnhancementOptions } from "@/lib/image-processing"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface ImageProcessorProps {
  className?: string
}

export default function ImageProcessor({ className }: ImageProcessorProps) {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [enhancedUrl, setEnhancedUrl] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [processingTime, setProcessingTime] = useState<number>(0)
  const [abortController, setAbortController] = useState<AbortController | null>(null)

  // Cleanup preview URLs when component unmounts
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
    }
  }, [previewUrl])

  // Timer for processing time
  useEffect(() => {
    let interval: NodeJS.Timeout | null = null

    if (isProcessing) {
      const startTime = Date.now()
      interval = setInterval(() => {
        setProcessingTime(Math.floor((Date.now() - startTime) / 1000))
      }, 1000)
    } else if (interval) {
      clearInterval(interval)
      setProcessingTime(0)
    }

    return () => {
      if (interval) clearInterval(interval)
    }
  }, [isProcessing])

  // Default enhancement options - using simpler options for testing
  const enhancementOptions: EnhancementOptions = {
    hdrEnhancement: true,
    perspectiveCorrection: false,
    virtualStaging: false,
    skyReplacement: false,
    lawnEnhancement: false,
    objectRemoval: false,
    upscaling: false,
    enhancementLevel: 50,
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null)
    setEnhancedUrl(null)

    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]

      // Check if file is an image
      if (!selectedFile.type.startsWith("image/")) {
        setError("Please select an image file")
        return
      }

      // Check file size (limit to 5MB for faster processing)
      if (selectedFile.size > 5 * 1024 * 1024) {
        setError("Image size should be less than 5MB for faster processing")
        return
      }

      setFile(selectedFile)
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      setPreviewUrl(URL.createObjectURL(selectedFile))
    }
  }

  const cancelProcessing = () => {
    if (abortController) {
      abortController.abort()
      setAbortController(null)
    }
    setIsProcessing(false)
    setError("Image processing was cancelled")
  }

  const processImage = async () => {
    if (!file) {
      setError("Please select an image first")
      return
    }

    // Cancel any existing processing
    if (abortController) {
      abortController.abort()
    }

    // Create a new AbortController
    const controller = new AbortController()
    setAbortController(controller)

    setIsProcessing(true)
    setError(null)
    setProcessingTime(0)

    try {
      const formData = new FormData()
      formData.append("image", file)
      formData.append("options", JSON.stringify(enhancementOptions))

      // Set a timeout of 60 seconds
      const timeoutId = setTimeout(() => {
        controller.abort()
        throw new Error("Request timed out after 60 seconds")
      }, 60000)

      const response = await fetch("/api/enhance", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to process image")
      }

      if (!data.enhancedImage) {
        throw new Error("No enhanced image was returned")
      }

      setEnhancedUrl(data.enhancedImage)
    } catch (err) {
      console.error("Error processing image:", err)

      if (err instanceof Error) {
        if (err.name === "AbortError") {
          setError("Request was cancelled or timed out. The AI service might be busy, please try again later.")
        } else {
          setError(err.message || "An unknown error occurred")
        }
      } else {
        setError("An unknown error occurred")
      }
    } finally {
      setIsProcessing(false)
      setAbortController(null)
    }
  }

  return (
    <div className={`space-y-6 ${className}`}>
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Test Image Enhancement</h2>

        <div className="space-y-4">
          <div>
            <label htmlFor="test-image" className="block text-sm font-medium mb-2">
              Select an image to test
            </label>
            <input
              type="file"
              id="test-image"
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-rose-50 file:text-rose-700
                hover:file:bg-rose-100"
            />
            <p className="mt-1 text-xs text-gray-500">
              For best results, use a clear photo of a property (interior or exterior) under 5MB.
            </p>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {previewUrl && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <p className="text-sm font-medium mb-2">Original Image</p>
                  <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
                    <img
                      src={previewUrl || "/placeholder.svg"}
                      alt="Original"
                      className="w-full h-full object-contain"
                    />
                  </div>
                </div>

                <div>
                  <p className="text-sm font-medium mb-2">Enhanced Image</p>
                  <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center">
                    {isProcessing ? (
                      <div className="flex flex-col items-center gap-2">
                        <Loader2 className="h-8 w-8 animate-spin text-rose-500" />
                        <p className="text-sm text-gray-500">Processing image... ({processingTime}s)</p>
                        <Button variant="outline" size="sm" onClick={cancelProcessing} className="mt-2">
                          Cancel
                        </Button>
                      </div>
                    ) : enhancedUrl ? (
                      <img
                        src={enhancedUrl || "/placeholder.svg"}
                        alt="Enhanced"
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <p className="text-gray-500 text-sm">Click "Process Image" to see the enhanced version</p>
                    )}
                  </div>
                </div>
              </div>

              <div className="flex justify-between">
                <p className="text-xs text-gray-500">
                  Note: Image processing may take 15-30 seconds depending on the AI service load.
                </p>
                <div className="flex gap-2">
                  {enhancedUrl && (
                    <Button
                      variant="outline"
                      onClick={() => {
                        setEnhancedUrl(null)
                        setError(null)
                      }}
                      className="flex items-center gap-1"
                    >
                      <RefreshCw className="h-4 w-4" />
                      Reset
                    </Button>
                  )}
                  <Button
                    onClick={processImage}
                    disabled={isProcessing || !file}
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
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}

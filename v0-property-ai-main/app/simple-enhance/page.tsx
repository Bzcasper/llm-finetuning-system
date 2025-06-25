"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Loader2 } from "lucide-react"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { enhancementModels } from "@/lib/enhancement-models"
import Link from "next/link"

export default function SimpleEnhancePage() {
  const [selectedModelId, setSelectedModelId] = useState<string>("basic-enhancer")
  const [enhancementLevel, setEnhancementLevel] = useState<number>(50)
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [enhancedUrl, setEnhancedUrl] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingTime, setProcessingTime] = useState<number | null>(null)
  const [modelUsed, setModelUsed] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setError(null)
    setEnhancedUrl(null)

    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      setPreviewUrl(URL.createObjectURL(selectedFile))
    }
  }

  const processImage = async () => {
    if (!file) {
      setError("Please select an image first")
      return
    }

    setIsProcessing(true)
    setError(null)
    setEnhancedUrl(null)
    setProcessingTime(null)
    setModelUsed(null)

    try {
      // Prepare form data
      const formData = new FormData()
      formData.append("image", file)
      formData.append("modelId", selectedModelId)
      formData.append("enhancementLevel", enhancementLevel.toString())

      // Send to our API
      const response = await fetch("/api/enhance-simple", {
        method: "POST",
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to process image")
      }

      if (!data.enhancedImage) {
        throw new Error("No enhanced image was returned")
      }

      setEnhancedUrl(data.enhancedImage)
      setProcessingTime(data.processingTime)
      setModelUsed(data.modelUsed)
    } catch (err) {
      console.error("Error processing image:", err)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsProcessing(false)
    }
  }

  const selectedModel = enhancementModels.find((model) => model.id === selectedModelId)

  return (
    <div className="container max-w-4xl py-8">
      <div className="mb-6">
        <Link href="/dashboard">
          <Button variant="outline" size="sm">
            Back to Dashboard
          </Button>
        </Link>
      </div>

      <h1 className="text-2xl font-bold mb-4">Simple Property Enhancement</h1>
      <p className="text-gray-500 mb-6">Enhance your property photos with our tiered enhancement models.</p>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="space-y-6">
          <div className="border p-4 rounded-lg">
            <h2 className="text-lg font-medium mb-4">1. Select Enhancement Tier</h2>
            <RadioGroup value={selectedModelId} onValueChange={setSelectedModelId}>
              {enhancementModels && enhancementModels.length > 0 ? (
                enhancementModels.map((model) => (
                  <div key={model.id} className="flex items-start space-x-2">
                    <RadioGroupItem value={model.id} id={model.id} className="mt-1" />
                    <div className="grid gap-1.5">
                      <Label htmlFor={model.id} className="font-medium">
                        {model.name}
                        <span className="ml-2 text-xs px-2 py-0.5 bg-gray-100 rounded-full">{model.tier}</span>
                      </Label>
                      <p className="text-sm text-gray-500">{model.description}</p>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {model.features &&
                          model.features.map((feature) => (
                            <span key={feature} className="text-xs px-2 py-0.5 bg-rose-50 text-rose-700 rounded-full">
                              {feature}
                            </span>
                          ))}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-sm text-gray-500">No enhancement models available</div>
              )}
            </RadioGroup>
          </div>

          <div className="border p-4 rounded-lg">
            <h2 className="text-lg font-medium mb-4">2. Enhancement Level</h2>
            <div className="space-y-4">
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label htmlFor="enhancement-level">Enhancement Strength</Label>
                  <span className="text-sm">{enhancementLevel}%</span>
                </div>
                <Slider
                  id="enhancement-level"
                  min={0}
                  max={100}
                  step={5}
                  value={[enhancementLevel]}
                  onValueChange={(value) => setEnhancementLevel(value[0])}
                />
                <div className="flex justify-between text-xs text-gray-500">
                  <span>Subtle</span>
                  <span>Balanced</span>
                  <span>Dramatic</span>
                </div>
              </div>
            </div>
          </div>

          <div className="border p-4 rounded-lg">
            <h2 className="text-lg font-medium mb-4">3. Select an Image</h2>
            <input type="file" accept="image/*" onChange={handleFileChange} className="block w-full" />
          </div>

          <Button
            onClick={processImage}
            disabled={isProcessing || !file}
            className="w-full bg-rose-500 hover:bg-rose-600"
          >
            {isProcessing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              "Enhance Image"
            )}
          </Button>

          {error && (
            <div className="p-3 bg-red-50 text-red-700 rounded-md">
              <p className="font-medium">{error}</p>
            </div>
          )}
        </div>

        <div className="space-y-6">
          {previewUrl && (
            <div className="border p-4 rounded-lg">
              <h2 className="text-lg font-medium mb-2">Original Image</h2>
              <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
                <img src={previewUrl || "/placeholder.svg"} alt="Original" className="w-full h-full object-contain" />
              </div>
            </div>
          )}

          {enhancedUrl && (
            <div className="border p-4 rounded-lg">
              <h2 className="text-lg font-medium mb-2">Enhanced Image</h2>
              <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
                <img src={enhancedUrl || "/placeholder.svg"} alt="Enhanced" className="w-full h-full object-contain" />
              </div>
              {modelUsed && processingTime && (
                <div className="mt-2 text-xs text-gray-500">
                  Enhanced with {modelUsed} in {processingTime.toFixed(2)} seconds
                </div>
              )}
            </div>
          )}

          {selectedModel && (
            <div className="border p-4 rounded-lg bg-gray-50">
              <h2 className="text-lg font-medium mb-2">Selected Model Information</h2>
              <div className="space-y-2">
                <div>
                  <span className="font-medium">Name:</span> {selectedModel.name}
                </div>
                <div>
                  <span className="font-medium">Tier:</span> {selectedModel.tier}
                </div>
                <div>
                  <span className="font-medium">Features:</span> {selectedModel.features.join(", ")}
                </div>
                <div>
                  <span className="font-medium">Cost per image:</span> ~{selectedModel.costPerImage} cents
                </div>
                <div>
                  <span className="font-medium">Description:</span> {selectedModel.description}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

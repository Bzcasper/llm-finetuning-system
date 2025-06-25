"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Loader2, CreditCard, ImageIcon, CheckCircle, AlertCircle } from "lucide-react"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { enhancementModels, type EnhancementCategory } from "@/lib/enhancement-models"
import BatchUploader from "@/components/batch-uploader"
import Link from "next/link"

// Demo user ID (in a real app, this would come from authentication)
const DEMO_USER_ID = "demo-user"

export default function ProEnhancePage() {
  // User credits
  const [credits, setCredits] = useState<number>(0)
  const [isLoadingCredits, setIsLoadingCredits] = useState(false)

  // Enhancement options
  const [selectedTab, setSelectedTab] = useState<string>("single")
  const [selectedModelId, setSelectedModelId] = useState<string>("basic-enhancer")
  const [enhancementLevel, setEnhancementLevel] = useState<number>(50)
  const [selectedCategories, setSelectedCategories] = useState<EnhancementCategory[]>(["basic"])
  const [backgroundPrompt, setBackgroundPrompt] = useState<string>("")

  // Image processing
  const [file, setFile] = useState<File | null>(null)
  const [batchFiles, setBatchFiles] = useState<File[]>([])
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [enhancedUrl, setEnhancedUrl] = useState<string | null>(null)
  const [batchResults, setBatchResults] = useState<{ [key: string]: string | null }>({})
  const [isProcessing, setIsProcessing] = useState(false)
  const [processingTime, setProcessingTime] = useState<number | null>(null)
  const [modelUsed, setModelUsed] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Load user credits on mount
  useEffect(() => {
    loadUserCredits()
  }, [])

  const loadUserCredits = async () => {
    setIsLoadingCredits(true)
    try {
      const response = await fetch(`/api/credits?userId=${DEMO_USER_ID}`)
      const data = await response.json()

      if (data.success) {
        setCredits(data.credits)
      } else {
        console.error("Error loading credits:", data.error)
      }
    } catch (error) {
      console.error("Error loading credits:", error)
    } finally {
      setIsLoadingCredits(false)
    }
  }

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

  const handleBatchFilesSelected = (files: File[]) => {
    setBatchFiles(files)
    setError(null)
    setBatchResults({})
  }

  const handleCategoryChange = (category: EnhancementCategory, checked: boolean) => {
    setSelectedCategories((prev) => {
      if (checked) {
        return [...prev, category]
      } else {
        return prev.filter((c) => c !== category)
      }
    })
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
      formData.append("userId", DEMO_USER_ID)
      formData.append("image", file)
      formData.append("modelId", selectedModelId)
      formData.append("enhancementLevel", enhancementLevel.toString())

      if (backgroundPrompt) {
        formData.append("backgroundPrompt", backgroundPrompt)
      }

      // Send to our API
      const response = await fetch("/api/enhance-pro", {
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
      setCredits(data.remainingCredits)
    } catch (err) {
      console.error("Error processing image:", err)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsProcessing(false)
    }
  }

  const processBatch = async () => {
    if (batchFiles.length === 0) {
      setError("Please select at least one image")
      return
    }

    setIsProcessing(true)
    setError(null)
    setBatchResults({})

    try {
      // Prepare form data
      const formData = new FormData()
      formData.append("userId", DEMO_USER_ID)
      formData.append("isBatch", "true")
      formData.append("modelId", selectedModelId)
      formData.append("enhancementLevel", enhancementLevel.toString())

      if (backgroundPrompt) {
        formData.append("backgroundPrompt", backgroundPrompt)
      }

      // Add all files
      batchFiles.forEach((file) => {
        formData.append("images", file)
      })

      // Send to our API
      const response = await fetch("/api/enhance-pro", {
        method: "POST",
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to process images")
      }

      if (!data.results) {
        throw new Error("No results were returned")
      }

      // Process results
      const processedResults: { [key: string]: string | null } = {}

      for (const [id, result] of Object.entries(data.results)) {
        if (result.success && result.enhancedImage) {
          processedResults[id] = result.enhancedImage
        } else {
          processedResults[id] = null
        }
      }

      setBatchResults(processedResults)
      setCredits(data.remainingCredits)
    } catch (err) {
      console.error("Error processing batch:", err)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsProcessing(false)
    }
  }

  const addCredits = async (amount: number) => {
    try {
      const response = await fetch("/api/credits", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          userId: DEMO_USER_ID,
          amount,
          description: "Added credits via dashboard",
        }),
      })

      const data = await response.json()

      if (data.success) {
        setCredits(data.credits)
        return true
      } else {
        console.error("Error adding credits:", data.error)
        return false
      }
    } catch (error) {
      console.error("Error adding credits:", error)
      return false
    }
  }

  // Get models for the selected categories
  const availableModels = enhancementModels.filter((model) => selectedCategories.includes(model.category))

  // Get the selected model
  const selectedModel = enhancementModels.find((model) => model.id === selectedModelId)

  return (
    <div className="container max-w-6xl py-8">
      <div className="mb-6">
        <Link href="/dashboard">
          <Button variant="outline" size="sm">
            Back to Dashboard
          </Button>
        </Link>
      </div>

      <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
        <div>
          <h1 className="text-2xl font-bold">Professional Property Enhancement</h1>
          <p className="text-gray-500">Enhance your property photos with our advanced AI models</p>
        </div>

        <Card className="w-full md:w-auto">
          <CardContent className="p-4 flex items-center gap-3">
            <CreditCard className="h-5 w-5 text-rose-500" />
            <div>
              <p className="text-sm font-medium">Your Credits</p>
              {isLoadingCredits ? (
                <div className="flex items-center gap-1">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  <span className="text-xs text-gray-500">Loading...</span>
                </div>
              ) : (
                <p className="text-xl font-bold">{credits}</p>
              )}
            </div>
            <Button size="sm" variant="outline" className="ml-2" onClick={() => addCredits(10)}>
              Add Credits
            </Button>
          </CardContent>
        </Card>
      </div>

      <Tabs value={selectedTab} onValueChange={setSelectedTab} className="w-full">
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="single">Single Image</TabsTrigger>
          <TabsTrigger value="batch">Batch Processing</TabsTrigger>
        </TabsList>

        <div className="grid md:grid-cols-3 gap-6 mt-6">
          <div className="md:col-span-1 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Enhancement Options</CardTitle>
                <CardDescription>Select the options for your enhancement</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-4">
                  <div>
                    <h3 className="text-sm font-medium mb-2">Enhancement Categories</h3>
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="category-basic"
                          checked={selectedCategories.includes("basic")}
                          onCheckedChange={(checked) => handleCategoryChange("basic", checked as boolean)}
                        />
                        <Label htmlFor="category-basic">Basic Enhancement</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="category-upscaling"
                          checked={selectedCategories.includes("upscaling")}
                          onCheckedChange={(checked) => handleCategoryChange("upscaling", checked as boolean)}
                        />
                        <Label htmlFor="category-upscaling">Upscaling</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="category-deblur"
                          checked={selectedCategories.includes("deblur")}
                          onCheckedChange={(checked) => handleCategoryChange("deblur", checked as boolean)}
                        />
                        <Label htmlFor="category-deblur">Deblur</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="category-background-removal"
                          checked={selectedCategories.includes("background-removal")}
                          onCheckedChange={(checked) => handleCategoryChange("background-removal", checked as boolean)}
                        />
                        <Label htmlFor="category-background-removal">Background Removal</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="category-background-replacement"
                          checked={selectedCategories.includes("background-replacement")}
                          onCheckedChange={(checked) =>
                            handleCategoryChange("background-replacement", checked as boolean)
                          }
                        />
                        <Label htmlFor="category-background-replacement">Background Replacement</Label>
                      </div>
                    </div>
                  </div>

                  {availableModels.length > 0 && (
                    <div>
                      <h3 className="text-sm font-medium mb-2">Select Model</h3>
                      <RadioGroup value={selectedModelId} onValueChange={setSelectedModelId} className="space-y-3">
                        {availableModels.map((model) => (
                          <div key={model.id} className="flex items-start space-x-2">
                            <RadioGroupItem value={model.id} id={model.id} className="mt-1" />
                            <div className="grid gap-1">
                              <Label htmlFor={model.id} className="font-medium">
                                {model.name}
                                <span className="ml-2 text-xs px-2 py-0.5 bg-gray-100 rounded-full">
                                  {model.creditCost} credits
                                </span>
                              </Label>
                              <p className="text-xs text-gray-500">{model.description}</p>
                            </div>
                          </div>
                        ))}
                      </RadioGroup>
                    </div>
                  )}

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

                  {selectedCategories.includes("background-replacement") && (
                    <div className="space-y-2">
                      <Label htmlFor="background-prompt">Background Prompt</Label>
                      <Input
                        id="background-prompt"
                        placeholder="e.g., blue sky with clouds"
                        value={backgroundPrompt}
                        onChange={(e) => setBackgroundPrompt(e.target.value)}
                      />
                      <p className="text-xs text-gray-500">Describe the background you want to replace with</p>
                    </div>
                  )}
                </div>
              </CardContent>
              <CardFooter>
                {selectedModel && (
                  <div className="w-full p-3 bg-gray-50 rounded-md text-sm">
                    <p className="font-medium">Selected: {selectedModel.name}</p>
                    <p className="text-xs text-gray-500 mt-1">Cost: {selectedModel.creditCost} credits per image</p>
                  </div>
                )}
              </CardFooter>
            </Card>
          </div>

          <div className="md:col-span-2 space-y-6">
            <TabsContent value="single" className="space-y-6 mt-0">
              <Card>
                <CardHeader>
                  <CardTitle>Upload Image</CardTitle>
                  <CardDescription>Select a single image to enhance</CardDescription>
                </CardHeader>
                <CardContent>
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="block w-full text-sm text-gray-500
                      file:mr-4 file:py-2 file:px-4
                      file:rounded-md file:border-0
                      file:text-sm file:font-semibold
                      file:bg-rose-50 file:text-rose-700
                      hover:file:bg-rose-100"
                  />
                </CardContent>
              </Card>

              <div className="grid md:grid-cols-2 gap-6">
                {previewUrl && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Original Image</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
                        <img
                          src={previewUrl || "/placeholder.svg"}
                          alt="Original"
                          className="w-full h-full object-contain"
                        />
                      </div>
                    </CardContent>
                  </Card>
                )}

                <Card>
                  <CardHeader>
                    <CardTitle>Enhanced Result</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center">
                      {isProcessing ? (
                        <div className="flex flex-col items-center gap-2">
                          <Loader2 className="h-8 w-8 animate-spin text-rose-500" />
                          <p className="text-sm text-gray-500">Processing image...</p>
                        </div>
                      ) : enhancedUrl ? (
                        <img
                          src={enhancedUrl || "/placeholder.svg"}
                          alt="Enhanced"
                          className="w-full h-full object-contain"
                        />
                      ) : (
                        <div className="text-center p-4">
                          <ImageIcon className="h-12 w-12 text-gray-300 mx-auto mb-2" />
                          <p className="text-gray-500">Enhanced image will appear here</p>
                        </div>
                      )}
                    </div>

                    {enhancedUrl && modelUsed && processingTime && (
                      <div className="mt-2 text-xs text-gray-500">
                        Enhanced with {modelUsed} in {processingTime.toFixed(2)} seconds
                      </div>
                    )}
                  </CardContent>
                  <CardFooter>
                    <Button
                      onClick={processImage}
                      disabled={isProcessing || !file || !selectedModel || credits < selectedModel.creditCost}
                      className="w-full bg-rose-500 hover:bg-rose-600"
                    >
                      {isProcessing ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Processing...
                        </>
                      ) : (
                        <>
                          Enhance Image
                          {selectedModel && <span className="ml-2 text-xs">({selectedModel.creditCost} credits)</span>}
                        </>
                      )}
                    </Button>
                  </CardFooter>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="batch" className="space-y-6 mt-0">
              <Card>
                <CardHeader>
                  <CardTitle>Batch Upload</CardTitle>
                  <CardDescription>Upload multiple images or a ZIP file containing images</CardDescription>
                </CardHeader>
                <CardContent>
                  <BatchUploader onFilesSelected={handleBatchFilesSelected} />
                </CardContent>
                <CardFooter>
                  <Button
                    onClick={processBatch}
                    disabled={
                      isProcessing ||
                      batchFiles.length === 0 ||
                      !selectedModel ||
                      credits < selectedModel.creditCost * batchFiles.length
                    }
                    className="w-full bg-rose-500 hover:bg-rose-600"
                  >
                    {isProcessing ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Processing Batch...
                      </>
                    ) : (
                      <>
                        Process {batchFiles.length} Images
                        {selectedModel && batchFiles.length > 0 && (
                          <span className="ml-2 text-xs">
                            ({selectedModel.creditCost * batchFiles.length} credits total)
                          </span>
                        )}
                      </>
                    )}
                  </Button>
                </CardFooter>
              </Card>

              {Object.keys(batchResults).length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle>Batch Results</CardTitle>
                    <CardDescription>{Object.keys(batchResults).length} images processed</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                      {Object.entries(batchResults).map(([id, result]) => (
                        <div key={id} className="space-y-1">
                          <div className="aspect-square rounded-lg border bg-gray-50 overflow-hidden">
                            {result ? (
                              <img
                                src={result || "/placeholder.svg"}
                                alt={`Result ${id}`}
                                className="h-full w-full object-cover"
                              />
                            ) : (
                              <div className="h-full w-full flex flex-col items-center justify-center p-4">
                                <AlertCircle className="h-8 w-8 text-red-500 mb-2" />
                                <p className="text-xs text-center text-gray-500">Failed to process</p>
                              </div>
                            )}
                          </div>
                          <p className="text-xs truncate">{id}</p>
                          {result ? (
                            <div className="flex items-center text-xs text-green-600">
                              <CheckCircle className="h-3 w-3 mr-1" />
                              Success
                            </div>
                          ) : (
                            <div className="flex items-center text-xs text-red-600">
                              <AlertCircle className="h-3 w-3 mr-1" />
                              Failed
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                  <CardFooter>
                    <Button variant="outline" className="w-full" onClick={() => window.print()}>
                      Download All Results
                    </Button>
                  </CardFooter>
                </Card>
              )}
            </TabsContent>
          </div>
        </div>
      </Tabs>

      {error && (
        <div className="mt-6 p-4 bg-red-50 text-red-700 rounded-md">
          <div className="flex items-start">
            <AlertCircle className="h-5 w-5 mr-2 mt-0.5" />
            <div>
              <p className="font-medium">Error</p>
              <p>{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

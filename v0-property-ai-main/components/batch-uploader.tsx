"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Upload, X, FileArchive, ImageIcon } from "lucide-react"
import JSZip from "jszip"

interface BatchUploaderProps {
  onFilesSelected: (files: File[]) => void
  maxFiles?: number
  acceptedTypes?: string
  className?: string
}

export default function BatchUploader({
  onFilesSelected,
  maxFiles = 50,
  acceptedTypes = "image/*,.zip",
  className = "",
}: BatchUploaderProps) {
  const [files, setFiles] = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessingZip, setIsProcessingZip] = useState(false)
  const [zipProgress, setZipProgress] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files) {
      await processFiles(Array.from(e.dataTransfer.files))
    }
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      await processFiles(Array.from(e.target.files))
    }
  }

  const processFiles = async (selectedFiles: File[]) => {
    const imageFiles: File[] = []
    const zipFiles: File[] = []

    // Separate image and zip files
    for (const file of selectedFiles) {
      if (file.type.startsWith("image/")) {
        imageFiles.push(file)
      } else if (file.type === "application/zip" || file.name.endsWith(".zip")) {
        zipFiles.push(file)
      }
    }

    // Process zip files if any
    if (zipFiles.length > 0) {
      setIsProcessingZip(true)
      setZipProgress(0)

      try {
        const extractedImages: File[] = []

        for (const zipFile of zipFiles) {
          const zip = new JSZip()
          const zipContent = await zip.loadAsync(zipFile)
          const totalFiles = Object.keys(zipContent.files).length
          let processedFiles = 0

          for (const [filename, zipEntry] of Object.entries(zipContent.files)) {
            if (!zipEntry.dir && filename.match(/\.(jpe?g|png|gif|webp|bmp)$/i)) {
              const content = await zipEntry.async("blob")
              const imageFile = new File([content], filename, {
                type: getImageMimeType(filename),
              })
              extractedImages.push(imageFile)
            }

            processedFiles++
            setZipProgress(Math.round((processedFiles / totalFiles) * 100))
          }
        }

        // Add extracted images to the image files
        imageFiles.push(...extractedImages)
      } catch (error) {
        console.error("Error processing zip file:", error)
        alert("Error processing zip file. Please try again.")
      } finally {
        setIsProcessingZip(false)
      }
    }

    // Limit the number of files
    const limitedFiles = imageFiles.slice(0, maxFiles)

    // Update state and call the callback
    setFiles((prev) => {
      const newFiles = [...prev, ...limitedFiles]
      // Limit to maxFiles
      const limitedNewFiles = newFiles.slice(0, maxFiles)
      // Call the callback
      onFilesSelected(limitedNewFiles)
      return limitedNewFiles
    })
  }

  const getImageMimeType = (filename: string): string => {
    const ext = filename.split(".").pop()?.toLowerCase()
    switch (ext) {
      case "jpg":
      case "jpeg":
        return "image/jpeg"
      case "png":
        return "image/png"
      case "gif":
        return "image/gif"
      case "webp":
        return "image/webp"
      case "bmp":
        return "image/bmp"
      default:
        return "image/jpeg"
    }
  }

  const removeFile = (index: number) => {
    setFiles((prev) => {
      const newFiles = prev.filter((_, i) => i !== index)
      onFilesSelected(newFiles)
      return newFiles
    })
  }

  const clearFiles = () => {
    setFiles([])
    onFilesSelected([])
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  return (
    <div className={`space-y-6 ${className}`}>
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center ${
          isDragging ? "border-rose-500 bg-rose-50" : "border-gray-300"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center justify-center space-y-4">
          <div className="rounded-full bg-rose-100 p-3">
            <Upload className="h-6 w-6 text-rose-500" />
          </div>
          <div className="space-y-2">
            <h3 className="text-lg font-medium">Drag and drop your images or ZIP file</h3>
            <p className="text-sm text-gray-500">or click to browse from your computer</p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            id="batch-file-upload"
            className="sr-only"
            multiple
            accept={acceptedTypes}
            onChange={handleFileChange}
          />
          <Button variant="outline" onClick={() => fileInputRef.current?.click()}>
            Browse Files
          </Button>
          <p className="text-xs text-gray-500">Supported formats: JPG, PNG, WEBP, ZIP. Max {maxFiles} files.</p>
        </div>
      </div>

      {isProcessingZip && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <FileArchive className="h-5 w-5 text-rose-500" />
            <p className="text-sm font-medium">Extracting images from ZIP file...</p>
          </div>
          <Progress value={zipProgress} className="h-2" />
        </div>
      )}

      {files.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-medium">Uploaded Images ({files.length})</h3>
            <Button variant="outline" size="sm" onClick={clearFiles}>
              Clear All
            </Button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {files.map((file, index) => (
              <div key={index} className="relative group">
                <div className="aspect-square rounded-lg border bg-gray-50 overflow-hidden">
                  <div className="h-full w-full flex items-center justify-center">
                    {file.type.startsWith("image/") ? (
                      <img
                        src={URL.createObjectURL(file) || "/placeholder.svg"}
                        alt={file.name}
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <ImageIcon className="h-8 w-8 text-gray-400" />
                    )}
                  </div>
                </div>
                <button
                  className="absolute -top-2 -right-2 bg-white rounded-full p-1 shadow-md opacity-0 group-hover:opacity-100 transition-opacity"
                  onClick={() => removeFile(index)}
                >
                  <X className="h-4 w-4" />
                </button>
                <p className="mt-1 text-xs truncate">{file.name}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

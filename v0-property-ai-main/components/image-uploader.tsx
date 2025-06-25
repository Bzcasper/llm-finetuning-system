"use client"

import type React from "react"

import { useState } from "react"
import { Upload, X, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function ImageUploader() {
  const [files, setFiles] = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    if (e.dataTransfer.files) {
      const newFiles = Array.from(e.dataTransfer.files).filter((file) => file.type.startsWith("image/"))
      setFiles((prev) => [...prev, ...newFiles])
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files).filter((file) => file.type.startsWith("image/"))
      setFiles((prev) => [...prev, ...newFiles])
    }
  }

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-6">
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
            <h3 className="text-lg font-medium">Drag and drop your images</h3>
            <p className="text-sm text-gray-500">or click to browse from your computer</p>
          </div>
          <input
            type="file"
            id="file-upload"
            className="sr-only"
            multiple
            accept="image/*"
            onChange={handleFileChange}
          />
          <Button variant="outline" onClick={() => document.getElementById("file-upload")?.click()}>
            Browse Files
          </Button>
          <p className="text-xs text-gray-500">Supported formats: JPG, PNG, WEBP. Max file size: 10MB</p>
        </div>
      </div>

      {files.length > 0 && (
        <div className="space-y-4">
          <h3 className="font-medium">Uploaded Images ({files.length})</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {files.map((file, index) => (
              <div key={index} className="relative group">
                <div className="aspect-square rounded-lg border bg-gray-50 overflow-hidden">
                  <div className="h-full w-full flex items-center justify-center">
                    {URL.createObjectURL(file) ? (
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
          <div className="flex justify-end space-x-2">
            <Button variant="outline" onClick={() => setFiles([])}>
              Clear All
            </Button>
            <Button className="bg-rose-500 hover:bg-rose-600">Continue to Enhancement</Button>
          </div>
        </div>
      )}
    </div>
  )
}

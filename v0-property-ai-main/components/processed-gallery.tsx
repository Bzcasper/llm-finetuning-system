"use client"

import { useState } from "react"
import { Download, Share2, Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

// Mock data for demonstration
const mockProcessedImages = [
  {
    id: 1,
    original: "/placeholder.svg?height=600&width=800",
    processed: "/placeholder.svg?height=600&width=800",
    name: "Living Room",
    enhancements: ["HDR", "Virtual Staging"],
  },
  {
    id: 2,
    original: "/placeholder.svg?height=600&width=800",
    processed: "/placeholder.svg?height=600&width=800",
    name: "Kitchen",
    enhancements: ["HDR", "Perspective Correction"],
  },
  {
    id: 3,
    original: "/placeholder.svg?height=600&width=800",
    processed: "/placeholder.svg?height=600&width=800",
    name: "Exterior Front",
    enhancements: ["Sky Replacement", "Lawn Enhancement"],
  },
  {
    id: 4,
    original: "/placeholder.svg?height=600&width=800",
    processed: "/placeholder.svg?height=600&width=800",
    name: "Master Bedroom",
    enhancements: ["HDR", "Virtual Staging"],
  },
]

export default function ProcessedGallery() {
  const [selectedImage, setSelectedImage] = useState<number | null>(null)

  return (
    <div className="space-y-6">
      <Tabs defaultValue="grid" className="w-full">
        <div className="flex justify-between items-center mb-4">
          <TabsList>
            <TabsTrigger value="grid">Grid</TabsTrigger>
            <TabsTrigger value="comparison">Comparison</TabsTrigger>
          </TabsList>
          <div className="flex gap-2">
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Download All
            </Button>
            <Button variant="outline" size="sm">
              <Share2 className="h-4 w-4 mr-2" />
              Share
            </Button>
          </div>
        </div>

        <TabsContent value="grid">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {mockProcessedImages.map((image) => (
              <div key={image.id} className="group relative cursor-pointer" onClick={() => setSelectedImage(image.id)}>
                <div
                  className={`aspect-square rounded-lg border overflow-hidden ${
                    selectedImage === image.id ? "ring-2 ring-rose-500" : ""
                  }`}
                >
                  <img
                    src={image.processed || "/placeholder.svg"}
                    alt={image.name}
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all flex items-center justify-center opacity-0 group-hover:opacity-100">
                  <Button variant="secondary" size="sm" className="shadow-lg">
                    View
                  </Button>
                </div>
                <div className="mt-2">
                  <p className="font-medium text-sm">{image.name}</p>
                  <p className="text-xs text-gray-500">{image.enhancements.join(", ")}</p>
                </div>
              </div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="comparison">
          <div className="space-y-8">
            {mockProcessedImages.map((image) => (
              <div key={image.id} className="space-y-2">
                <div className="flex justify-between items-center">
                  <h3 className="font-medium">{image.name}</h3>
                  <div className="flex gap-2">
                    <Button variant="ghost" size="sm">
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="sm">
                      <Trash2 className="h-4 w-4 text-red-500" />
                    </Button>
                  </div>
                </div>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-gray-500 mb-1">Original</p>
                    <div className="aspect-video rounded-lg border overflow-hidden">
                      <img
                        src={image.original || "/placeholder.svg"}
                        alt={`Original ${image.name}`}
                        className="h-full w-full object-cover"
                      />
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 mb-1">Enhanced</p>
                    <div className="aspect-video rounded-lg border overflow-hidden">
                      <img
                        src={image.processed || "/placeholder.svg"}
                        alt={`Enhanced ${image.name}`}
                        className="h-full w-full object-cover"
                      />
                    </div>
                  </div>
                </div>
                <p className="text-xs text-gray-500">Enhancements: {image.enhancements.join(", ")}</p>
              </div>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

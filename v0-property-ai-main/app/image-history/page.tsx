"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Loader2, Download } from "lucide-react"
import Link from "next/link"
import { format } from "date-fns"

type BlobImageRecord = {
  id: string
  originalFilename: string
  enhancedImageUrl: string
  originalImageUrl?: string
  userId: string
  modelUsed: string
  processingTime: number
  prompt?: string
  enhancementLevel?: number
  requestId?: string
  createdAt: string
  blobPath: string
}

export default function ImageHistoryPage() {
  const [records, setRecords] = useState<BlobImageRecord[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState<"all" | "date" | "user">("all")
  const [filterValue, setFilterValue] = useState<string>("")

  useEffect(() => {
    loadImageRecords()
  }, [filter, filterValue])

  const loadImageRecords = async () => {
    setIsLoading(true)
    setError(null)

    try {
      let url = "/api/images/records"

      if (filter === "user" && filterValue) {
        url += `?userId=${encodeURIComponent(filterValue)}`
      } else if (filter === "date" && filterValue) {
        url += `?date=${encodeURIComponent(filterValue)}`
      }

      const response = await fetch(url)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to load image records")
      }

      setRecords(data.records || [])
    } catch (err) {
      console.error("Error loading image records:", err)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  const formatDate = (dateString: string) => {
    try {
      return format(new Date(dateString), "PPP 'at' p")
    } catch (error) {
      return dateString
    }
  }

  const downloadImage = async (url: string, filename: string) => {
    try {
      const response = await fetch(url)
      const blob = await response.blob()
      const downloadUrl = window.URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = downloadUrl
      link.download = filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(downloadUrl)
    } catch (error) {
      console.error("Error downloading image:", error)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-8">
          <Link href="/dashboard">
            <Button variant="outline" size="sm" className="mb-4">
              ← Back to Dashboard
            </Button>
          </Link>

          <div className="text-center mb-8">
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-4">Image Processing History</h1>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              View all processed images organized by date and user, stored securely in Vercel Blob.
            </p>
          </div>
        </div>

        {/* Filters */}
        <div className="bg-white rounded-xl shadow-sm border p-6 mb-8">
          <div className="flex flex-wrap gap-4 items-end">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-sm font-medium text-gray-700 mb-2">Filter by</label>
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value as "all" | "date" | "user")}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">All Images</option>
                <option value="date">By Date</option>
                <option value="user">By User</option>
              </select>
            </div>

            {filter === "date" && (
              <div className="flex-1 min-w-[200px]">
                <label className="block text-sm font-medium text-gray-700 mb-2">Select Date</label>
                <input
                  type="date"
                  value={filterValue}
                  onChange={(e) => setFilterValue(e.target.value)}
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            )}

            {filter === "user" && (
              <div className="flex-1 min-w-[200px]">
                <label className="block text-sm font-medium text-gray-700 mb-2">User ID</label>
                <input
                  type="text"
                  value={filterValue}
                  onChange={(e) => setFilterValue(e.target.value)}
                  placeholder="Enter user ID"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            )}

            <Button onClick={() => loadImageRecords()} className="bg-blue-600 hover:bg-blue-700 px-6 py-3">
              Apply Filter
            </Button>
          </div>
        </div>

        {/* Content */}
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="text-center">
              <Loader2 className="h-12 w-12 animate-spin text-blue-600 mx-auto mb-4" />
              <p className="text-gray-600">Loading image records...</p>
            </div>
          </div>
        ) : error ? (
          <div className="bg-red-50 border border-red-200 rounded-xl p-6">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              </div>
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800">Error loading image records</h3>
                <p className="text-sm text-red-700 mt-1">{error}</p>
              </div>
            </div>
          </div>
        ) : records.length === 0 ? (
          <div className="bg-gray-50 border border-gray-200 rounded-xl p-12 text-center">
            <div className="mx-auto h-12 w-12 text-gray-400 mb-4">
              <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">No image records found</h3>
            <p className="text-gray-600">Try adjusting your filters or process some images first.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {records.map((record) => (
              <div
                key={record.id}
                className="bg-white rounded-xl shadow-sm border overflow-hidden hover:shadow-md transition-shadow"
              >
                <div className="aspect-square bg-gray-100 relative group">
                  <img
                    src={record.enhancedImageUrl || "/placeholder.svg"}
                    alt={record.originalFilename}
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-200 flex items-center justify-center">
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex gap-2">
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={() => downloadImage(record.enhancedImageUrl, `${record.originalFilename}-enhanced`)}
                        className="bg-white/90 hover:bg-white"
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </div>

                <div className="p-4">
                  <h3 className="font-semibold text-gray-900 truncate mb-2">{record.originalFilename}</h3>

                  <div className="space-y-1 text-sm text-gray-600 mb-4">
                    <p>
                      <span className="font-medium">Processed:</span> {formatDate(record.createdAt)}
                    </p>
                    <p>
                      <span className="font-medium">Model:</span> {record.modelUsed}
                    </p>
                    <p>
                      <span className="font-medium">User:</span> {record.userId}
                    </p>
                    {record.processingTime && (
                      <p>
                        <span className="font-medium">Time:</span> {record.processingTime.toFixed(2)}s
                      </p>
                    )}
                    {record.prompt && (
                      <p className="truncate">
                        <span className="font-medium">Prompt:</span> {record.prompt}
                      </p>
                    )}
                  </div>

                  <div className="flex gap-2">
                    <Button size="sm" variant="outline" asChild className="flex-1">
                      <a href={record.enhancedImageUrl} target="_blank" rel="noopener noreferrer">
                        View Enhanced
                      </a>
                    </Button>
                    {record.originalImageUrl && (
                      <Button size="sm" variant="outline" asChild className="flex-1">
                        <a href={record.originalImageUrl} target="_blank" rel="noopener noreferrer">
                          Original
                        </a>
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

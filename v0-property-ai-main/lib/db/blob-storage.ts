import { put, del, list } from "@vercel/blob"
import { v4 as uuidv4 } from "uuid"

// Define the image record type for blob storage
export type BlobImageRecord = {
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
  metadata?: Record<string, any>
  blobPath: string
}

// Convert base64 to blob
function base64ToBlob(base64: string, mimeType = "image/png"): Blob {
  const byteCharacters = atob(base64.split(",")[1])
  const byteNumbers = new Array(byteCharacters.length)

  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i)
  }

  const byteArray = new Uint8Array(byteNumbers)
  return new Blob([byteArray], { type: mimeType })
}

// Save an image record to Vercel Blob
export async function saveBlobImageRecord(
  record: Omit<BlobImageRecord, "id" | "createdAt" | "blobPath">,
): Promise<BlobImageRecord> {
  const id = uuidv4()
  const timestamp = new Date().toISOString()
  const date = timestamp.split("T")[0] // YYYY-MM-DD format

  // Create the blob path structure
  const blobPath = `images/${date}/${record.userId}/${id}`

  let enhancedImageUrl = record.enhancedImageUrl
  let originalImageUrl = record.originalImageUrl

  try {
    // Upload enhanced image if it's base64
    if (record.enhancedImageUrl.startsWith("data:")) {
      const enhancedBlob = base64ToBlob(record.enhancedImageUrl)
      const enhancedResult = await put(`${blobPath}-enhanced.png`, enhancedBlob, {
        access: "public",
        addRandomSuffix: false,
      })
      enhancedImageUrl = enhancedResult.url
    }

    // Upload original image if it's base64
    if (record.originalImageUrl && record.originalImageUrl.startsWith("data:")) {
      const originalBlob = base64ToBlob(record.originalImageUrl)
      const originalResult = await put(`${blobPath}-original.png`, originalBlob, {
        access: "public",
        addRandomSuffix: false,
      })
      originalImageUrl = originalResult.url
    }

    // Create the full record
    const fullRecord: BlobImageRecord = {
      ...record,
      id,
      createdAt: timestamp,
      blobPath,
      enhancedImageUrl,
      originalImageUrl,
    }

    // Save metadata as JSON blob
    const metadataBlob = new Blob([JSON.stringify(fullRecord, null, 2)], {
      type: "application/json",
    })

    await put(`${blobPath}-metadata.json`, metadataBlob, {
      access: "public",
      addRandomSuffix: false,
    })

    return fullRecord
  } catch (error) {
    console.error("Error saving image record to blob storage:", error)
    throw new Error("Failed to save image record")
  }
}

// Get all image records from Vercel Blob
export async function getAllBlobImageRecords(): Promise<BlobImageRecord[]> {
  try {
    const { blobs } = await list({
      prefix: "images/",
    })

    const records: BlobImageRecord[] = []

    // Filter for metadata files and fetch their content
    const metadataBlobs = blobs.filter((blob) => blob.pathname.endsWith("-metadata.json"))

    for (const metadataBlob of metadataBlobs) {
      try {
        const response = await fetch(metadataBlob.url)
        const record = await response.json()
        records.push(record)
      } catch (error) {
        console.error(`Error fetching metadata from ${metadataBlob.url}:`, error)
      }
    }

    // Sort by creation date (newest first)
    return records.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
  } catch (error) {
    console.error("Error getting image records from blob storage:", error)
    return []
  }
}

// Get image records for a specific user
export async function getUserBlobImageRecords(userId: string): Promise<BlobImageRecord[]> {
  try {
    const { blobs } = await list({
      prefix: `images/`,
    })

    const records: BlobImageRecord[] = []

    // Filter for metadata files that belong to the user
    const userMetadataBlobs = blobs.filter(
      (blob) => blob.pathname.includes(`/${userId}/`) && blob.pathname.endsWith("-metadata.json"),
    )

    for (const metadataBlob of userMetadataBlobs) {
      try {
        const response = await fetch(metadataBlob.url)
        const record = await response.json()
        records.push(record)
      } catch (error) {
        console.error(`Error fetching user metadata from ${metadataBlob.url}:`, error)
      }
    }

    return records.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
  } catch (error) {
    console.error("Error getting user image records from blob storage:", error)
    return []
  }
}

// Get image records for a specific date
export async function getBlobImageRecordsByDate(date: string): Promise<BlobImageRecord[]> {
  try {
    const { blobs } = await list({
      prefix: `images/${date}/`,
    })

    const records: BlobImageRecord[] = []

    // Filter for metadata files
    const metadataBlobs = blobs.filter((blob) => blob.pathname.endsWith("-metadata.json"))

    for (const metadataBlob of metadataBlobs) {
      try {
        const response = await fetch(metadataBlob.url)
        const record = await response.json()
        records.push(record)
      } catch (error) {
        console.error(`Error fetching date metadata from ${metadataBlob.url}:`, error)
      }
    }

    return records.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
  } catch (error) {
    console.error("Error getting date image records from blob storage:", error)
    return []
  }
}

// Delete an image record from Vercel Blob
export async function deleteBlobImageRecord(id: string): Promise<boolean> {
  try {
    const { blobs } = await list({
      prefix: "images/",
    })

    // Find all blobs related to this record
    const recordBlobs = blobs.filter((blob) => blob.pathname.includes(`/${id}-`))

    // Delete all related blobs
    for (const blob of recordBlobs) {
      try {
        await del(blob.url)
      } catch (error) {
        console.error(`Error deleting blob ${blob.url}:`, error)
      }
    }

    return true
  } catch (error) {
    console.error("Error deleting image record from blob storage:", error)
    return false
  }
}

// Get a specific image record by ID
export async function getBlobImageRecord(id: string): Promise<BlobImageRecord | null> {
  try {
    const { blobs } = await list({
      prefix: "images/",
    })

    // Find the metadata file for this record
    const metadataBlob = blobs.find((blob) => blob.pathname.includes(`/${id}-metadata.json`))

    if (!metadataBlob) {
      return null
    }

    const response = await fetch(metadataBlob.url)
    const record = await response.json()
    return record
  } catch (error) {
    console.error("Error getting image record from blob storage:", error)
    return null
  }
}

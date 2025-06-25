import fs from "fs"
import path from "path"
import { v4 as uuidv4 } from "uuid"

// Define the image record type
export type ImageRecord = {
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
  folderPath: string
}

// Base directory for storing image records
const BASE_DIR = path.join(process.cwd(), "data", "images")

// Ensure the base directory exists
export function ensureDirectoryExists(directory: string): void {
  if (!fs.existsSync(directory)) {
    fs.mkdirSync(directory, { recursive: true })
  }
}

// Initialize the storage
export function initStorage(): void {
  ensureDirectoryExists(BASE_DIR)
}

// Create a folder path based on date and user
export function createFolderPath(userId: string): string {
  const date = new Date()
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")

  const folderPath = path.join(BASE_DIR, `${year}-${month}-${day}`, userId)
  ensureDirectoryExists(folderPath)

  return folderPath
}

// Save an image record to the database
export async function saveImageRecord(
  record: Omit<ImageRecord, "id" | "createdAt" | "folderPath">,
): Promise<ImageRecord> {
  // Create a unique ID for the record
  const id = uuidv4()

  // Create the folder path
  const folderPath = createFolderPath(record.userId)

  // Create the full record
  const fullRecord: ImageRecord = {
    ...record,
    id,
    createdAt: new Date().toISOString(),
    folderPath,
  }

  // Save the record to a JSON file
  const recordPath = path.join(folderPath, `${id}.json`)
  fs.writeFileSync(recordPath, JSON.stringify(fullRecord, null, 2))

  // If the enhanced image is a base64 string, save it as a file
  if (record.enhancedImageUrl.startsWith("data:")) {
    const imageData = record.enhancedImageUrl.split(",")[1]
    const imageBuffer = Buffer.from(imageData, "base64")
    const imagePath = path.join(folderPath, `${id}-enhanced.png`)
    fs.writeFileSync(imagePath, imageBuffer)

    // Update the record with the file path
    fullRecord.enhancedImageUrl = `/api/images/${id}-enhanced.png`
    fs.writeFileSync(recordPath, JSON.stringify(fullRecord, null, 2))
  }

  // If the original image is a base64 string, save it as a file
  if (record.originalImageUrl && record.originalImageUrl.startsWith("data:")) {
    const imageData = record.originalImageUrl.split(",")[1]
    const imageBuffer = Buffer.from(imageData, "base64")
    const imagePath = path.join(folderPath, `${id}-original.png`)
    fs.writeFileSync(imagePath, imageBuffer)

    // Update the record with the file path
    fullRecord.originalImageUrl = `/api/images/${id}-original.png`
    fs.writeFileSync(recordPath, JSON.stringify(fullRecord, null, 2))
  }

  return fullRecord
}

// Get an image record by ID
export async function getImageRecord(id: string): Promise<ImageRecord | null> {
  // Search for the record in all folders
  const records = await getAllImageRecords()
  return records.find((record) => record.id === id) || null
}

// Get all image records
export async function getAllImageRecords(): Promise<ImageRecord[]> {
  const records: ImageRecord[] = []

  // Ensure the base directory exists
  ensureDirectoryExists(BASE_DIR)

  // Recursively search for JSON files in all folders
  const searchDirectory = (directory: string) => {
    const files = fs.readdirSync(directory)

    for (const file of files) {
      const filePath = path.join(directory, file)
      const stats = fs.statSync(filePath)

      if (stats.isDirectory()) {
        searchDirectory(filePath)
      } else if (file.endsWith(".json")) {
        try {
          const record = JSON.parse(fs.readFileSync(filePath, "utf-8"))
          records.push(record)
        } catch (error) {
          console.error(`Error reading record file ${filePath}:`, error)
        }
      }
    }
  }

  searchDirectory(BASE_DIR)

  // Sort records by creation date (newest first)
  return records.sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
}

// Get image records for a specific user
export async function getUserImageRecords(userId: string): Promise<ImageRecord[]> {
  const allRecords = await getAllImageRecords()
  return allRecords.filter((record) => record.userId === userId)
}

// Get image records for a specific date
export async function getImageRecordsByDate(date: string): Promise<ImageRecord[]> {
  const allRecords = await getAllImageRecords()
  return allRecords.filter((record) => record.createdAt.startsWith(date))
}

// Delete an image record
export async function deleteImageRecord(id: string): Promise<boolean> {
  const record = await getImageRecord(id)

  if (!record) {
    return false
  }

  // Delete the JSON file
  const recordPath = path.join(record.folderPath, `${id}.json`)
  if (fs.existsSync(recordPath)) {
    fs.unlinkSync(recordPath)
  }

  // Delete the enhanced image file
  if (record.enhancedImageUrl && record.enhancedImageUrl.includes(id)) {
    const enhancedImagePath = path.join(record.folderPath, `${id}-enhanced.png`)
    if (fs.existsSync(enhancedImagePath)) {
      fs.unlinkSync(enhancedImagePath)
    }
  }

  // Delete the original image file
  if (record.originalImageUrl && record.originalImageUrl.includes(id)) {
    const originalImagePath = path.join(record.folderPath, `${id}-original.png`)
    if (fs.existsSync(originalImagePath)) {
      fs.unlinkSync(originalImagePath)
    }
  }

  return true
}

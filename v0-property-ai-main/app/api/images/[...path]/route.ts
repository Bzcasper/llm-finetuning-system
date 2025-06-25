import { type NextRequest, NextResponse } from "next/server"
import fs from "fs"
import path from "path"

export async function GET(request: NextRequest, { params }: { params: { path: string[] } }) {
  try {
    // Get the image path from the URL
    const imagePath = params.path.join("/")

    // Construct the full path to the image
    const fullPath = path.join(process.cwd(), "data", "images", imagePath)

    // Check if the file exists
    if (!fs.existsSync(fullPath)) {
      return NextResponse.json({ error: "Image not found" }, { status: 404 })
    }

    // Read the file
    const fileBuffer = fs.readFileSync(fullPath)

    // Determine the content type based on the file extension
    const extension = path.extname(fullPath).toLowerCase()
    let contentType = "image/png" // Default content type

    if (extension === ".jpg" || extension === ".jpeg") {
      contentType = "image/jpeg"
    } else if (extension === ".gif") {
      contentType = "image/gif"
    } else if (extension === ".webp") {
      contentType = "image/webp"
    }

    // Return the image
    return new NextResponse(fileBuffer, {
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=31536000, immutable",
      },
    })
  } catch (error) {
    console.error("Error serving image:", error)
    return NextResponse.json({ error: "Failed to serve image" }, { status: 500 })
  }
}

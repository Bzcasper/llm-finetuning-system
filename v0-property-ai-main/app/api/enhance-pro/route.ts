import { type NextRequest, NextResponse } from "next/server"
import { enhanceImage, enhanceBatch } from "@/lib/enhancement-service"
import { getUserCredits } from "@/lib/credit-system"

// Set a longer timeout for the API route
export const maxDuration = 60 // 60 seconds (maximum allowed)

export async function POST(request: NextRequest) {
  console.log("API: Starting professional enhancement request")

  try {
    // Check if FAL_KEY is set
    const falKey = process.env.FAL_KEY || ""
    console.log(`API: Environment check - FAL_KEY present: ${!!falKey}, length: ${falKey.length}`)

    if (!falKey) {
      return NextResponse.json({ error: "FAL_KEY environment variable is not set" }, { status: 500 })
    }

    // Parse the form data
    const formData = await request.formData()

    // Get user ID (in a real app, this would come from authentication)
    const userId = (formData.get("userId") as string) || "demo-user"

    // Check if this is a batch request
    const isBatch = formData.get("isBatch") === "true"

    // Get model ID and options
    const modelId = formData.get("modelId") as string
    const enhancementLevelStr = (formData.get("enhancementLevel") as string) || "50"
    const backgroundPrompt = (formData.get("backgroundPrompt") as string) || ""

    // Parse enhancement level
    const enhancementLevel = Number.parseInt(enhancementLevelStr, 10) || 50

    // Get user's current credits
    const userCredits = await getUserCredits(userId)

    // Prepare enhancement options
    const options = {
      enhancementLevel,
      backgroundPrompt: backgroundPrompt || undefined,
    }

    if (isBatch) {
      // Handle batch processing
      const imageFiles = formData.getAll("images") as File[]

      if (!imageFiles || imageFiles.length === 0) {
        return NextResponse.json({ error: "No image files provided" }, { status: 400 })
      }

      console.log(`API: Received ${imageFiles.length} images for batch processing`)

      // Convert all files to base64
      const images = await Promise.all(
        imageFiles.map(async (file, index) => {
          const bytes = await file.arrayBuffer()
          const buffer = Buffer.from(bytes)
          const base64 = `data:${file.type};base64,${buffer.toString("base64")}`
          return {
            id: file.name || `image-${index}`,
            base64,
          }
        }),
      )

      // Process the batch
      const results = await enhanceBatch(userId, images, modelId, options)

      return NextResponse.json({
        success: true,
        results,
        remainingCredits: await getUserCredits(userId),
      })
    } else {
      // Handle single image processing
      const imageFile = formData.get("image") as File

      if (!imageFile) {
        return NextResponse.json({ error: "No image file provided" }, { status: 400 })
      }

      console.log("API: Received single image for processing")

      // Convert the file to base64
      const bytes = await imageFile.arrayBuffer()
      const buffer = Buffer.from(bytes)
      const base64Image = `data:${imageFile.type};base64,${buffer.toString("base64")}`

      // Process the image
      const result = await enhanceImage(userId, base64Image, modelId, options)

      if (!result.success) {
        return NextResponse.json({ error: result.error }, { status: 500 })
      }

      return NextResponse.json({
        success: true,
        enhancedImage: result.enhancedImage,
        modelUsed: result.modelUsed,
        processingTime: result.processingTime,
        creditsCost: result.creditsCost,
        remainingCredits: await getUserCredits(userId),
      })
    }
  } catch (error) {
    console.error("API: Error enhancing image:", error)

    if (error instanceof Error) {
      return NextResponse.json(
        {
          error: `Failed to enhance image: ${error.message}`,
        },
        { status: 500 },
      )
    }

    return NextResponse.json(
      {
        error: "Failed to enhance image",
        details: String(error),
      },
      { status: 500 },
    )
  }
}

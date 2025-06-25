import { type NextRequest, NextResponse } from "next/server"
import { enhanceImage } from "@/lib/fal-client"

// Set a timeout for the API route
export const maxDuration = 60 // 60 seconds (maximum allowed)

export async function POST(request: NextRequest) {
  console.log("API: Starting simple enhancement request")

  try {
    // Check if FAL_KEY is set
    const falKey = process.env.FAL_KEY || ""
    console.log(`API: Environment check - FAL_KEY present: ${!!falKey}, length: ${falKey.length}`)

    if (!falKey) {
      return NextResponse.json({ error: "FAL_KEY environment variable is not set" }, { status: 500 })
    }

    // Parse the form data
    const formData = await request.formData()
    const imageFile = formData.get("image") as File
    const modelId = (formData.get("modelId") as string) || "fast-sdxl"
    const enhancementLevelStr = (formData.get("enhancementLevel") as string) || "50"
    const userId = (formData.get("userId") as string) || "anonymous"

    console.log("API: Received form data", {
      hasImageFile: !!imageFile,
      imageType: imageFile?.type,
      imageSize: imageFile?.size,
      modelId,
      enhancementLevel: enhancementLevelStr,
      userId,
    })

    if (!imageFile) {
      return NextResponse.json({ error: "No image file provided" }, { status: 400 })
    }

    // Convert the file to base64
    const bytes = await imageFile.arrayBuffer()
    const buffer = Buffer.from(bytes)
    const base64Image = `data:${imageFile.type};base64,${buffer.toString("base64")}`

    console.log("API: Image converted to base64, starting enhancement")

    // Parse enhancement level
    const enhancementLevel = Number.parseInt(enhancementLevelStr, 10) || 50

    // Process the image using our enhancement service
    const result = await enhanceImage(base64Image, modelId, { enhancementLevel }, userId, imageFile.name)

    if (!result.success) {
      console.error("API: Enhancement failed:", result.error)
      return NextResponse.json({ error: result.error }, { status: 500 })
    }

    console.log(`API: Enhancement request submitted successfully`)

    // If the request was queued for webhook processing
    if (result.webhookQueued) {
      return NextResponse.json({
        success: true,
        webhookQueued: true,
        requestId: result.requestId,
        modelUsed: result.modelUsed,
      })
    }

    // If the result contains an enhanced image (synchronous processing)
    return NextResponse.json({
      success: true,
      enhancedImage: result.enhancedImage,
      modelUsed: result.modelUsed,
      processingTime: result.processingTime,
    })
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

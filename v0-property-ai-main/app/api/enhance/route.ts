import { type NextRequest, NextResponse } from "next/server"
import { enhanceImage, type EnhancementOptions } from "@/lib/fal-client"
import { withCreditCheck } from "@/lib/auth/middleware"
import { Session } from "next-auth"

// Set a longer timeout for the API route
export const maxDuration = 60 // 60 seconds

export const POST = withCreditCheck(
  async (request: NextRequest, session: Session) => {
    console.log(`API: Starting image enhancement request for user ${session.user.id}`)

    try {
      // Log environment variables status (safely)
      const falKey = process.env.FAL_KEY || ""
      console.log(`API: Environment check - FAL_KEY present: ${!!falKey}, length: ${falKey.length}`)

      if (!falKey) {
        return NextResponse.json(
          { 
            error: "Server configuration error",
            details: "FAL_KEY environment variable is not set",
            timestamp: new Date().toISOString(),
          }, 
          { status: 500 }
        )
      }

      const formData = await request.formData()
      const imageFile = formData.get("image") as File
      const enhancementOptionsStr = formData.get("options") as string

      console.log("API: Received form data", {
        userId: session.user.id,
        hasImageFile: !!imageFile,
        imageType: imageFile?.type,
        imageSize: imageFile?.size,
        hasOptions: !!enhancementOptionsStr,
      })

      // Validate input
      if (!imageFile) {
        return NextResponse.json(
          { 
            error: "No image file provided",
            timestamp: new Date().toISOString(),
          }, 
          { status: 400 }
        )
      }

      // Check file size (max 10MB)
      if (imageFile.size > 10 * 1024 * 1024) {
        return NextResponse.json(
          { 
            error: "File too large",
            details: "Maximum file size is 10MB",
            timestamp: new Date().toISOString(),
          }, 
          { status: 400 }
        )
      }

      // Check file type
      const allowedTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
      if (!allowedTypes.includes(imageFile.type)) {
        return NextResponse.json(
          { 
            error: "Invalid file type",
            details: "Only JPEG, PNG, and WebP images are supported",
            timestamp: new Date().toISOString(),
          }, 
          { status: 400 }
        )
      }

      let enhancementOptions: EnhancementOptions
      try {
        enhancementOptions = JSON.parse(enhancementOptionsStr) as EnhancementOptions
      } catch (e) {
        console.error("API: Failed to parse enhancement options", e)
        return NextResponse.json(
          { 
            error: "Invalid enhancement options",
            details: "Enhancement options must be valid JSON",
            timestamp: new Date().toISOString(),
          }, 
          { status: 400 }
        )
      }

      // Convert the file to base64
      const bytes = await imageFile.arrayBuffer()
      const buffer = Buffer.from(bytes)
      const base64Image = `data:${imageFile.type};base64,${buffer.toString("base64")}`

      console.log("API: Image converted to base64, starting AI processing")
      console.log("API: Base64 image format:", base64Image.substring(0, 50) + "...")

      // Process the image using our utility function
      const startTime = Date.now()
      const result = await enhanceImage(base64Image, enhancementOptions)
      const processingTime = (Date.now() - startTime) / 1000

      console.log(`API: Image processing completed in ${processingTime.toFixed(2)} seconds for user ${session.user.id}`)

      if (!result || !result.base64) {
        console.error("API: No result or base64 image returned from enhanceImage")
        return NextResponse.json(
          { 
            error: "Failed to generate enhanced image",
            details: "AI processing service returned empty result",
            timestamp: new Date().toISOString(),
          }, 
          { status: 500 }
        )
      }

      return NextResponse.json({
        success: true,
        enhancedImage: result.base64,
        requestId: result.requestId,
        processingTime,
        userId: session.user.id,
        timestamp: new Date().toISOString(),
      })
    } catch (error) {
      console.error("API: Error enhancing image:", error)

      // Provide more specific error messages
      if (error instanceof Error) {
        console.error("API: Error name:", error.name)
        console.error("API: Error message:", error.message)
        console.error("API: Error stack:", error.stack)

        if (error.message.includes("environment variables")) {
          return NextResponse.json(
            {
              error: "Server configuration error",
              details: "Please check that the FAL_KEY environment variable is set correctly",
              timestamp: new Date().toISOString(),
            },
            { status: 500 },
          )
        }

        return NextResponse.json(
          {
            error: "Failed to enhance image",
            details: error.message,
            timestamp: new Date().toISOString(),
          },
          { status: 500 },
        )
      }

      return NextResponse.json(
        {
          error: "Failed to enhance image",
          details: String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 },
      )
    }
  },
  5 // Credit cost for enhancement
)

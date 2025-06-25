import { type NextRequest, NextResponse } from "next/server"

// Set a timeout for the API route
export const maxDuration = 60 // 60 seconds (maximum allowed)

export async function POST(request: NextRequest) {
  console.log("Direct API: Starting direct Fal API test")

  try {
    // Log environment variables status (safely)
    const falKey = process.env.FAL_KEY || ""
    console.log(`Direct API: Environment check - FAL_KEY present: ${!!falKey}, length: ${falKey.length}`)

    if (!falKey) {
      return NextResponse.json({ error: "FAL_KEY environment variable is not set" }, { status: 500 })
    }

    const body = await request.json()
    const { image } = body

    if (!image) {
      return NextResponse.json({ error: "No image provided in request body" }, { status: 400 })
    }

    console.log("Direct API: Received image data, sending to Fal API directly")

    // Make a direct request to the Fal API
    try {
      const response = await fetch("https://gateway.fal.ai/direct/fast-sdxl/image-to-image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Key ${falKey}`,
        },
        body: JSON.stringify({
          image_data: image,
          prompt: "Enhance this real estate photo to make it look professional",
          strength: 0.3,
          guidance_scale: 5.0,
        }),
      })

      console.log(`Direct API: Fal API response status: ${response.status}`)

      if (!response.ok) {
        const errorText = await response.text()
        console.error("Direct API: Fal API error response:", errorText)

        return NextResponse.json(
          {
            error: `Fal API returned an error: ${response.status} ${response.statusText}`,
            details: errorText,
          },
          { status: response.status },
        )
      }

      const data = await response.json()
      console.log("Direct API: Successfully received response from Fal API")

      return NextResponse.json({
        success: true,
        enhancedImage: data.image_data || data.image,
      })
    } catch (error) {
      console.error("Direct API: Error making request to Fal API:", error)

      if (error instanceof Error) {
        return NextResponse.json(
          {
            error: `Error connecting to Fal API: ${error.message}`,
            details: error.stack,
          },
          { status: 500 },
        )
      }

      throw error
    }
  } catch (error) {
    console.error("Direct API: Unexpected error:", error)

    return NextResponse.json({ error: "An unexpected error occurred" }, { status: 500 })
  }
}

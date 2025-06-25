import { type NextRequest, NextResponse } from "next/server"
import { generateExampleImage } from "@/lib/fal-client"

// Set a timeout for the API route
export const maxDuration = 60 // 60 seconds (maximum allowed)

export async function POST(request: NextRequest) {
  console.log("Generate Image API: Starting")

  try {
    // Check if FAL_KEY is set
    const falKey = process.env.FAL_KEY || ""
    console.log(`Generate Image API: FAL_KEY present: ${!!falKey}, length: ${falKey.length}`)

    if (!falKey) {
      return NextResponse.json({ error: "FAL_KEY environment variable is not set" }, { status: 500 })
    }

    // Get the request body
    const body = await request.json()
    const { prompt } = body

    if (!prompt) {
      return NextResponse.json({ error: "Missing prompt" }, { status: 400 })
    }

    console.log(`Generate Image API: Received prompt: ${prompt}`)

    // Generate the image using our client
    const result = await generateExampleImage(prompt)

    if (!result.success) {
      return NextResponse.json({ error: result.error }, { status: 500 })
    }

    return NextResponse.json({
      success: true,
      imageUrl: result.imageUrl,
      requestId: result.requestId,
    })
  } catch (error) {
    console.error("Generate Image API: Unexpected error:", error)

    if (error instanceof Error) {
      return NextResponse.json(
        {
          error: `An unexpected error occurred: ${error.message}`,
          details: error.stack,
        },
        { status: 500 },
      )
    }

    return NextResponse.json(
      {
        error: "An unexpected error occurred",
        details: String(error),
      },
      { status: 500 },
    )
  }
}

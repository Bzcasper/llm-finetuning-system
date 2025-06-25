import { type NextRequest, NextResponse } from "next/server"

// Set a timeout for the API route
export const maxDuration = 60 // 60 seconds (maximum allowed)

export async function POST(request: NextRequest) {
  console.log("REST Generate API: Starting")

  try {
    // Get the FAL_KEY from environment variables
    const falKey = process.env.FAL_KEY

    if (!falKey) {
      console.error("REST Generate API: Missing FAL_KEY environment variable")
      return NextResponse.json({ error: "FAL_KEY environment variable is not set" }, { status: 500 })
    }

    console.log(`REST Generate API: FAL_KEY present, length: ${falKey.length}`)

    // Get the request body
    const body = await request.json()
    const { prompt } = body

    if (!prompt) {
      console.error("REST Generate API: Missing prompt in request body")
      return NextResponse.json({ error: "Missing prompt" }, { status: 400 })
    }

    console.log(`REST Generate API: Received prompt: ${prompt}`)

    // Make a direct request to the Fal AI API using the REST approach
    console.log("REST Generate API: Making request to Fal AI using fast-sdxl model")

    try {
      const falResponse = await fetch("https://gateway.fal.ai/direct/fast-sdxl", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Key ${falKey}`,
        },
        body: JSON.stringify({
          prompt: prompt,
          negative_prompt: "ugly, deformed, disfigured, poor quality, low quality",
        }),
      })

      console.log(`REST Generate API: Fal AI response status: ${falResponse.status}`)

      // Get the response from Fal AI
      const falData = await falResponse.json()

      if (!falResponse.ok) {
        console.error("REST Generate API: Fal AI error:", falData)
        return NextResponse.json(
          {
            error: `Fal AI returned an error: ${falResponse.status} ${falResponse.statusText}`,
            details: falData,
          },
          { status: falResponse.status },
        )
      }

      console.log("REST Generate API: Received successful response from Fal AI")
      console.log("REST Generate API: Response keys:", Object.keys(falData))

      // Check if the response contains an image
      if (!falData.image_url && !falData.images) {
        console.error("REST Generate API: No image in Fal AI response")
        return NextResponse.json(
          {
            error: "No image was returned by Fal AI",
            details: falData,
          },
          { status: 500 },
        )
      }

      // Return the image URL
      const imageUrl = falData.image_url || (falData.images && falData.images[0])
      console.log(`REST Generate API: Returning image URL: ${imageUrl.substring(0, 50)}...`)

      return NextResponse.json({
        success: true,
        imageUrl: imageUrl,
        requestId: falData.request_id || falData.requestId,
      })
    } catch (falError) {
      console.error("REST Generate API: Error making request to Fal AI:", falError)

      if (falError instanceof Error) {
        return NextResponse.json(
          {
            error: `Error connecting to Fal AI: ${falError.message}`,
            details: falError.stack,
          },
          { status: 500 },
        )
      }

      throw falError
    }
  } catch (error) {
    console.error("REST Generate API: Unexpected error:", error)

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

import { type NextRequest, NextResponse } from "next/server"

// Set a longer timeout for the API route
export const maxDuration = 60 // 60 seconds

export async function POST(request: NextRequest) {
  try {
    // Get the token from the query parameters
    const token = request.nextUrl.searchParams.get("token")

    if (!token) {
      return NextResponse.json({ error: "Missing token" }, { status: 400 })
    }

    // In a real app, you would validate the token against your database or cache
    // For this example, we'll just use it as an identifier

    // Get the FAL_KEY from environment variables
    const falKey = process.env.FAL_KEY

    if (!falKey) {
      return NextResponse.json({ error: "FAL_KEY environment variable is not set" }, { status: 500 })
    }

    // Get the request body
    const body = await request.json()

    // Extract the model ID and input from the request
    const { modelId, input } = body

    if (!modelId) {
      return NextResponse.json({ error: "Missing modelId" }, { status: 400 })
    }

    console.log(`Proxy: Forwarding request to Fal AI for model ${modelId}`)
    console.log(`Proxy: Input:`, JSON.stringify(input))

    // Forward the request to Fal AI
    const falResponse = await fetch(`https://gateway.fal.ai/direct/${modelId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Key ${falKey}`,
      },
      body: JSON.stringify(input),
    })

    // Get the response from Fal AI
    const falData = await falResponse.json()

    if (!falResponse.ok) {
      console.error(`Proxy: Fal AI error:`, falData)
      return NextResponse.json(
        {
          error: `Fal AI returned an error: ${falResponse.status} ${falResponse.statusText}`,
          details: falData,
        },
        { status: falResponse.status },
      )
    }

    console.log(`Proxy: Received response from Fal AI`)

    // Return the response from Fal AI
    return NextResponse.json(falData)
  } catch (error) {
    console.error("Proxy: Error:", error)

    if (error instanceof Error) {
      return NextResponse.json(
        {
          error: `Proxy error: ${error.message}`,
        },
        { status: 500 },
      )
    }

    return NextResponse.json(
      {
        error: "An unexpected error occurred",
      },
      { status: 500 },
    )
  }
}

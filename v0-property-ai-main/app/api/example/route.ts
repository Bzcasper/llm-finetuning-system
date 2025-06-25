import { NextResponse } from "next/server"
import { runExampleCode } from "@/lib/fal-client"

// Set a timeout for the API route
export const maxDuration = 60 // 60 seconds (maximum allowed)

export async function POST() {
  console.log("Example API: Starting example code execution")

  try {
    // Log environment variables status (safely)
    const falKey = process.env.FAL_KEY || ""
    console.log(`Example API: Environment check - FAL_KEY present: ${!!falKey}, length: ${falKey.length}`)

    if (!falKey) {
      return NextResponse.json({ error: "FAL_KEY environment variable is not set" }, { status: 500 })
    }

    const result = await runExampleCode()

    console.log("Example API: Example code execution completed successfully")

    if (!result || !result.base64) {
      console.error("Example API: No result or base64 image returned")
      return NextResponse.json({ error: "Failed to generate image" }, { status: 500 })
    }

    return NextResponse.json({
      success: true,
      imageBase64: result.base64,
      requestId: result.requestId,
    })
  } catch (error) {
    console.error("Example API: Error running example:", error)

    if (error instanceof Error) {
      console.error("Example API: Error name:", error.name)
      console.error("Example API: Error message:", error.message)
      console.error("Example API: Error stack:", error.stack)

      return NextResponse.json(
        {
          error: `Failed to run example: ${error.message}`,
          stack: error.stack,
        },
        { status: 500 },
      )
    }

    return NextResponse.json(
      {
        error: "Failed to run example",
        details: String(error),
      },
      { status: 500 },
    )
  }
}

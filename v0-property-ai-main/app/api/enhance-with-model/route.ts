import { type NextRequest, NextResponse } from "next/server"
import { getModelById } from "@/lib/fal-models"

// Set a timeout for the API route
export const maxDuration = 60 // 60 seconds (maximum allowed)

export async function POST(request: NextRequest) {
  console.log("Enhance With Model API: Starting")

  try {
    // Get the FAL_KEY from environment variables
    const falKey = process.env.FAL_KEY

    if (!falKey) {
      console.error("Enhance With Model API: Missing FAL_KEY environment variable")
      return NextResponse.json({ error: "FAL_KEY environment variable is not set" }, { status: 500 })
    }

    console.log(`Enhance With Model API: FAL_KEY present, length: ${falKey.length}`)

    // Parse the form data
    const formData = await request.formData()
    const imageFile = formData.get("image") as File
    const modelId = formData.get("modelId") as string

    if (!imageFile) {
      console.error("Enhance With Model API: Missing image file")
      return NextResponse.json({ error: "Missing image file" }, { status: 400 })
    }

    if (!modelId) {
      console.error("Enhance With Model API: Missing model ID")
      return NextResponse.json({ error: "Missing model ID" }, { status: 400 })
    }

    // Get the model details
    const model = getModelById(modelId)
    if (!model) {
      console.error(`Enhance With Model API: Invalid model ID: ${modelId}`)
      return NextResponse.json({ error: `Invalid model ID: ${modelId}` }, { status: 400 })
    }

    console.log(`Enhance With Model API: Using model: ${model.name} (${model.endpoint})`)

    // Convert the file to base64
    const bytes = await imageFile.arrayBuffer()
    const buffer = Buffer.from(bytes)
    const base64Image = `data:${imageFile.type};base64,${buffer.toString("base64")}`

    // Prepare the request based on the model type
    const endpoint = `https://gateway.fal.ai/direct/${model.endpoint}`
    let requestBody: any = {}

    switch (model.type) {
      case "upscaling":
        // For upscaling models
        requestBody = {
          image_data: base64Image,
          scale: 2, // Default scale factor
        }
        break

      case "deblur":
        // For deblur models
        requestBody = {
          image_data: base64Image,
        }
        break

      case "background":
        // For background removal/replacement models
        if (model.id === "bria-background-remove") {
          requestBody = {
            image_data: base64Image,
          }
        } else if (model.id === "bria-background-replace") {
          requestBody = {
            image_data: base64Image,
            background_prompt: "bright blue sky with white clouds", // Default background for real estate
          }
        }
        break

      case "enhancement":
      default:
        // For general enhancement models
        requestBody = {
          image_data: base64Image,
        }
        break
    }

    console.log(`Enhance With Model API: Making request to ${endpoint}`)
    console.log(`Enhance With Model API: Request body keys: ${Object.keys(requestBody).join(", ")}`)

    // Make the request to Fal AI
    const falResponse = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Key ${falKey}`,
      },
      body: JSON.stringify(requestBody),
    })

    console.log(`Enhance With Model API: Fal AI response status: ${falResponse.status}`)

    // Parse the response
    const falData = await falResponse.json()

    if (!falResponse.ok) {
      console.error("Enhance With Model API: Fal AI error:", falData)
      return NextResponse.json(
        {
          error: `Fal AI returned an error: ${falResponse.status} ${falResponse.statusText}`,
          details: falData,
        },
        { status: falResponse.status },
      )
    }

    console.log("Enhance With Model API: Received successful response from Fal AI")
    console.log("Enhance With Model API: Response keys:", Object.keys(falData).join(", "))

    // Extract the enhanced image based on the model type
    let enhancedImage: string | null = null

    if (falData.image_data) {
      enhancedImage = falData.image_data
    } else if (falData.images && falData.images.length > 0) {
      enhancedImage = falData.images[0]
    } else if (falData.image) {
      enhancedImage = falData.image
    } else if (falData.output_image_data) {
      enhancedImage = falData.output_image_data
    } else if (falData.output_image) {
      enhancedImage = falData.output_image
    }

    if (!enhancedImage) {
      console.error("Enhance With Model API: No enhanced image in response:", falData)
      return NextResponse.json(
        {
          error: "No enhanced image was returned by Fal AI",
          details: falData,
        },
        { status: 500 },
      )
    }

    console.log("Enhance With Model API: Successfully extracted enhanced image")

    return NextResponse.json({
      success: true,
      enhancedImage,
      modelId,
      modelName: model.name,
    })
  } catch (error) {
    console.error("Enhance With Model API: Unexpected error:", error)

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

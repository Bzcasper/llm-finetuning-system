import { getModelById } from "./enhancement-models"
import { useCredits } from "./credit-system"

export type EnhancementOptions = {
  enhancementLevel?: number // 0-100
  backgroundPrompt?: string // For background replacement
  removeObjects?: string[] // For object removal
}

export type EnhancementResult = {
  success: boolean
  enhancedImage?: string
  modelUsed?: string
  processingTime?: number
  creditsCost?: number
  error?: string
}

export async function enhanceImage(
  userId: string,
  imageBase64: string,
  modelId: string,
  options: EnhancementOptions = {},
): Promise<EnhancementResult> {
  let hasEnoughCredits = false
  let model

  try {
    console.log("enhanceImage: Starting enhancement process")

    // Get the FAL_KEY from environment
    const falKey = process.env.FAL_KEY
    if (!falKey) {
      throw new Error("FAL_KEY environment variable is not set")
    }

    // Get the model
    model = getModelById(modelId)
    if (!model) {
      throw new Error(`Invalid model ID: ${modelId}`)
    }

    // Check if user has enough credits
    hasEnoughCredits = await useCredits(userId, model.creditCost, `Used ${model.name} enhancement`)

    if (!hasEnoughCredits) {
      throw new Error("Not enough credits to use this enhancement")
    }

    console.log(`enhanceImage: Using model: ${model.name} (${model.endpoint})`)

    // Prepare the request based on the model category
    const endpoint = `https://gateway.fal.ai/direct/${model.endpoint}`
    let requestBody: any = {}

    // Basic enhancement models
    if (model.category === "basic") {
      if (model.id === "basic-enhancer") {
        // Using fast-sdxl with low strength
        requestBody = {
          image_data: imageBase64,
          prompt: "Enhance this real estate photo to make it look professional",
          strength: options.enhancementLevel ? options.enhancementLevel / 200 : 0.3, // Convert 0-100 to 0-0.5
          guidance_scale: 5.0,
        }
      } else if (model.id === "thera") {
        // Thera model
        requestBody = {
          image_data: imageBase64,
        }
      }
    }
    // Upscaling models
    else if (model.category === "upscaling") {
      requestBody = {
        image_data: imageBase64,
        scale: 2, // Default scale factor
      }
    }
    // Deblur models
    else if (model.category === "deblur") {
      requestBody = {
        image_data: imageBase64,
      }
    }
    // Background removal
    else if (model.category === "background-removal") {
      requestBody = {
        image_data: imageBase64,
      }
    }
    // Background replacement
    else if (model.category === "background-replacement") {
      requestBody = {
        image_data: imageBase64,
        background_prompt: options.backgroundPrompt || "bright blue sky with white clouds", // Default background for real estate
      }
    }

    console.log(`enhanceImage: Making request to ${endpoint}`)

    const startTime = Date.now()

    // Make the request to Fal AI
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Key ${falKey}`,
      },
      body: JSON.stringify(requestBody),
    })

    const processingTime = (Date.now() - startTime) / 1000
    console.log(`enhanceImage: Request completed in ${processingTime.toFixed(2)} seconds`)

    if (!response.ok) {
      const errorData = await response.json()
      console.error("enhanceImage: Error response from Fal AI:", errorData)
      throw new Error(`Fal AI returned an error: ${response.status} ${response.statusText}`)
    }

    const data = await response.json()
    console.log("enhanceImage: Received successful response from Fal AI")

    // Extract the enhanced image based on the model
    let enhancedImage: string | null = null

    if (data.image_data) {
      enhancedImage = data.image_data
    } else if (data.images && data.images.length > 0) {
      enhancedImage = data.images[0]
    } else if (data.image) {
      enhancedImage = data.image
    } else if (data.output_image_data) {
      enhancedImage = data.output_image_data
    } else if (data.output_image) {
      enhancedImage = data.output_image
    }

    if (!enhancedImage) {
      throw new Error("No enhanced image was returned by Fal AI")
    }

    return {
      success: true,
      enhancedImage,
      modelUsed: model.name,
      processingTime,
      creditsCost: model.creditCost,
    }
  } catch (error) {
    console.error("enhanceImage: Error:", error)
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    }
  }
}

// Function to process multiple images (batch processing)
export async function enhanceBatch(
  userId: string,
  images: { id: string; base64: string }[],
  modelId: string,
  options: EnhancementOptions = {},
): Promise<{ [key: string]: EnhancementResult }> {
  const results: { [key: string]: EnhancementResult } = {}

  // Process images sequentially to avoid overwhelming the API
  for (const image of images) {
    results[image.id] = await enhanceImage(userId, image.base64, modelId, options)
  }

  return results
}

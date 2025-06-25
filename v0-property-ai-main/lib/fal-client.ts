import { fal } from "@fal-ai/client"

// Configure the client to use the proxy
fal.config({
  proxyUrl: "/api/fal/proxy",
})

// Example code from fal.ai documentation
export async function runExampleCode() {
  console.log("runExampleCode: Starting example code execution")

  try {
    // Create the model
    console.log("runExampleCode: Creating fal.image model")
    const model = fal.image("flux/dev")

    // Run the model
    console.log("runExampleCode: Running model with prompt: a cat")
    const result = await model({ prompt: "a cat" })

    console.log("runExampleCode: Model run completed successfully")

    // Check if the result contains an image
    if (!result || !result.images || result.images.length === 0) {
      console.error("runExampleCode: No image returned")
      throw new Error("No image was returned")
    }

    // Return the base64 representation of the image
    return {
      base64: result.images[0],
      requestId: result.request_id,
    }
  } catch (error) {
    console.error("runExampleCode: Error running example:", error)
    throw error
  }
}

export type EnhancementOptions = {
  enhancementLevel?: number // 0-100
  backgroundPrompt?: string // For background replacement
  removeObjects?: string[] // For object removal
}

export async function enhanceImage(
  imageBase64: string,
  modelId = "fal-ai/fast-sdxl",
  options: EnhancementOptions = {},
  userId = "anonymous",
  originalFilename = "unknown.jpg",
) {
  console.log("enhanceImage: Starting image enhancement with fal.ai client")

  try {
    // Construct a prompt based on enhancement options
    let prompt = "Enhance this real estate photo to make it look professional"

    if (options.enhancementLevel) {
      // Adjust prompt based on enhancement level
      if (options.enhancementLevel > 75) {
        prompt += ", with dramatic lighting and vibrant colors"
      } else if (options.enhancementLevel > 50) {
        prompt += ", with balanced lighting and natural colors"
      } else {
        prompt += ", with subtle improvements"
      }
    }

    console.log("enhanceImage: Using prompt:", prompt)

    // Make sure the image data is properly formatted
    let imageData = imageBase64
    if (!imageData.startsWith("data:image/")) {
      console.log("enhanceImage: Adding data:image prefix to base64 string")
      imageData = `data:image/jpeg;base64,${imageBase64}`
    }

    const startTime = Date.now()
    let result
    let modelName

    // Use the appropriate model based on the modelId
    if (modelId === "fast-sdxl" || modelId === "fal-ai/fast-sdxl") {
      // For image-to-image enhancement
      const model = fal.image("fast-sdxl/image-to-image")
      modelName = "Fast SDXL"

      result = await model({
        image_data: imageData,
        prompt: prompt,
        strength: options.enhancementLevel ? options.enhancementLevel / 200 : 0.3, // Convert 0-100 to 0-0.5
        guidance_scale: 5.0,
      })
    } else {
      // Default to flux model for other cases
      const model = fal.image("flux/dev")
      modelName = "Flux"

      result = await model({
        prompt: prompt,
        image_data: imageData,
        strength: options.enhancementLevel ? options.enhancementLevel / 100 : 0.5,
      })
    }

    const processingTime = (Date.now() - startTime) / 1000
    console.log(`enhanceImage: Image processing completed in ${processingTime.toFixed(2)} seconds`)

    if (!result || !result.images || result.images.length === 0) {
      console.error("enhanceImage: No image returned")
      throw new Error("No image was returned")
    }

    return {
      success: true,
      enhancedImage: result.images[0],
      modelUsed: modelName,
      processingTime,
    }
  } catch (error) {
    console.error("enhanceImage: Error in image enhancement:", error)

    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    }
  }
}

// Function to generate an example image
export async function generateExampleImage(prompt = "a beautiful house with a garden", userId = "anonymous") {
  try {
    console.log("generateExampleImage: Starting image generation with prompt:", prompt)

    const startTime = Date.now()

    const model = fal.image("flux/dev")
    const result = await model({
      prompt,
      image_size: "square_hd",
    })

    const processingTime = (Date.now() - startTime) / 1000
    console.log(`generateExampleImage: Image generation completed in ${processingTime.toFixed(2)} seconds`)

    if (!result || !result.images || result.images.length === 0) {
      console.error("generateExampleImage: No image returned")
      throw new Error("No image was returned")
    }

    return {
      success: true,
      imageUrl: result.images[0],
      requestId: result.request_id,
    }
  } catch (error) {
    console.error("generateExampleImage: Error:", error)
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error occurred",
    }
  }
}

import { experimental_generateImage as generateImage } from "ai"
import { fal } from "@ai-sdk/fal"
import { validateEnvVars } from "./config"

export type EnhancementOptions = {
  hdrEnhancement?: boolean
  perspectiveCorrection?: boolean
  virtualStaging?: boolean
  stagingStyle?: string
  skyReplacement?: boolean
  skyOption?: string
  lawnEnhancement?: boolean
  objectRemoval?: boolean
  upscaling?: boolean
  enhancementLevel?: number
}

export async function enhanceImage(base64Image: string, options: EnhancementOptions) {
  console.log("enhanceImage: Starting image enhancement")

  // Validate environment variables
  if (!validateEnvVars()) {
    console.error("enhanceImage: Missing required environment variables")
    throw new Error("Missing required environment variables")
  }

  // Construct a simpler prompt for testing
  let prompt = "Enhance this real estate photo to make it look professional"

  if (options.hdrEnhancement) {
    prompt += ", improve lighting and dynamic range"
  }

  console.log("enhanceImage: Using prompt:", prompt)

  // Generate enhanced image using Fal AI
  try {
    console.log("enhanceImage: Creating fal.image model")

    // Create the model instance correctly - using flux-lora model as in the example
    const model = fal.image("fal-ai/flux-lora")

    console.log("enhanceImage: Model created, starting image generation")

    // Following the example pattern exactly
    const { image } = await generateImage({
      model,
      prompt,
      images: [base64Image],
    })

    console.log("enhanceImage: Image generation completed successfully")

    // Return the base64 representation of the image
    return {
      base64: `data:image/png;base64,${Buffer.from(image.uint8Array).toString("base64")}`,
    }
  } catch (error) {
    console.error("enhanceImage: Error in image enhancement:", error)
    throw error
  }
}

// Collection of fal.ai models recommended for real estate image enhancement

export type FalModel = {
  id: string
  name: string
  description: string
  endpoint: string
  type: "enhancement" | "upscaling" | "background" | "deblur" | "other"
  commercial: boolean
}

export const falModels: Record<string, FalModel> = {
  "clarity-upscaler": {
    id: "clarity-upscaler",
    name: "Clarity Upscaler",
    description: "High-quality image upscaling for sharper, clearer real estate photos",
    endpoint: "fal-ai/clarity-upscaler",
    type: "upscaling",
    commercial: true,
  },
  swin2sr: {
    id: "swin2sr",
    name: "Swin2SR Upscaler",
    description: "Superior quality upscaling for sharper, clearer results",
    endpoint: "fal-ai/swin2sr",
    type: "upscaling",
    commercial: true,
  },
  ccsr: {
    id: "ccsr",
    name: "CCSR Upscaler",
    description: "State-of-the-art image upscaler for high-quality results",
    endpoint: "fal-ai/ccsr",
    type: "upscaling",
    commercial: true,
  },
  "nafnet-deblur": {
    id: "nafnet-deblur",
    name: "NAFNet Deblur",
    description: "Removes blur from images to improve clarity",
    endpoint: "fal-ai/nafnet/deblur",
    type: "deblur",
    commercial: true,
  },
  thera: {
    id: "thera",
    name: "Thera Enhancer",
    description: "Fixes low resolution images with fast speed and quality",
    endpoint: "fal-ai/thera",
    type: "enhancement",
    commercial: true,
  },
  "bria-background-remove": {
    id: "bria-background-remove",
    name: "Bria Background Removal",
    description: "Removes backgrounds from images (safe for commercial use)",
    endpoint: "fal-ai/bria/background/remove",
    type: "background",
    commercial: true,
  },
  "bria-background-replace": {
    id: "bria-background-replace",
    name: "Bria Background Replacement",
    description: "Replaces backgrounds in images via text prompts (safe for commercial use)",
    endpoint: "fal-ai/bria/background/replace",
    type: "background",
    commercial: true,
  },
  "fast-sdxl": {
    id: "fast-sdxl",
    name: "Fast SDXL",
    description: "General purpose image enhancement and generation",
    endpoint: "fast-sdxl",
    type: "enhancement",
    commercial: false,
  },
}

// Get models by type
export function getModelsByType(type: FalModel["type"]): FalModel[] {
  return Object.values(falModels).filter((model) => model.type === type)
}

// Get all commercial models
export function getCommercialModels(): FalModel[] {
  return Object.values(falModels).filter((model) => model.commercial)
}

// Get model by ID
export function getModelById(id: string): FalModel | undefined {
  return falModels[id]
}

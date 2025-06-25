// Ensure this is at the top of the file
export type EnhancementTier = "basic" | "premium" | "professional"

export type EnhancementCategory = "basic" | "upscaling" | "deblur" | "background-removal" | "background-replacement"

export type EnhancementModel = {
  id: string
  name: string
  endpoint: string
  description: string
  tier: EnhancementTier
  category: EnhancementCategory
  features: string[]
  costPerImage: number // approximate cost in cents
}

// Our carefully selected models for different tiers
export const enhancementModels: EnhancementModel[] = [
  {
    id: "basic-enhancer",
    name: "Basic Enhancer",
    endpoint: "fast-sdxl", // Using fast-sdxl with low strength for basic enhancement
    description: "Simple enhancement for lighting and color correction",
    tier: "basic",
    category: "basic",
    features: ["hdr"],
    costPerImage: 1, // ~1 cent per image
  },
  {
    id: "premium-enhancer",
    name: "Premium Enhancer",
    endpoint: "fal-ai/thera", // Thera for better quality enhancement
    description: "Advanced enhancement with improved clarity and detail",
    tier: "premium",
    category: "upscaling",
    features: ["hdr", "upscaling", "deblur"],
    costPerImage: 3, // ~3 cents per image
  },
  {
    id: "professional-enhancer",
    name: "Professional Enhancer",
    endpoint: "fal-ai/clarity-upscaler", // Clarity upscaler for best quality
    description: "Professional-grade enhancement with maximum quality",
    tier: "professional",
    category: "deblur",
    features: ["hdr", "upscaling", "deblur", "background-replacement"],
    costPerImage: 5, // ~5 cents per image
  },
]

// Get model by ID
export function getModelById(id: string): EnhancementModel | undefined {
  return enhancementModels.find((model) => model.id === id)
}

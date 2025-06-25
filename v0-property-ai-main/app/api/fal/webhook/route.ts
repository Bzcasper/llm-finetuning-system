import { type NextRequest, NextResponse } from "next/server"
import { saveBlobImageRecord } from "@/lib/db/blob-storage"

export async function POST(request: NextRequest) {
  try {
    // Parse the webhook payload
    const webhookData = await request.json()

    console.log("Received webhook data:", JSON.stringify(webhookData, null, 2))

    // Check if the webhook contains a valid payload
    if (webhookData.status !== "OK" || !webhookData.payload) {
      console.error("Invalid webhook payload:", webhookData)
      return NextResponse.json({ error: "Invalid webhook payload" }, { status: 400 })
    }

    // Extract the request ID
    const requestId = webhookData.request_id

    // Check if we have metadata for this request in our temporary storage
    // In a real application, you would store this in a database
    const metadata = global.webhookMetadata?.[requestId]

    if (!metadata) {
      console.warn(`No metadata found for request ID: ${requestId}`)
    }

    // Extract the image URL from the payload
    let imageUrl = ""
    if (webhookData.payload.images && webhookData.payload.images.length > 0) {
      imageUrl = webhookData.payload.images[0].url || webhookData.payload.images[0]
    } else if (webhookData.payload.image_data) {
      imageUrl = webhookData.payload.image_data
    } else if (webhookData.payload.image) {
      imageUrl = webhookData.payload.image
    }

    if (!imageUrl) {
      console.error("No image URL found in webhook payload:", webhookData)
      return NextResponse.json({ error: "No image URL found in payload" }, { status: 400 })
    }

    // Save the image record using blob storage
    const record = await saveBlobImageRecord({
      originalFilename: metadata?.originalFilename || "unknown.jpg",
      enhancedImageUrl: imageUrl,
      originalImageUrl: metadata?.originalImageUrl,
      userId: metadata?.userId || "anonymous",
      modelUsed: metadata?.modelUsed || "unknown",
      processingTime: metadata?.processingTime || 0,
      prompt: metadata?.prompt,
      enhancementLevel: metadata?.enhancementLevel,
      requestId,
      metadata: {
        ...metadata,
        webhookPayload: webhookData.payload,
      },
    })

    console.log("Saved image record to blob storage:", record.id)

    // Clean up the metadata
    if (global.webhookMetadata && global.webhookMetadata[requestId]) {
      delete global.webhookMetadata[requestId]
    }

    return NextResponse.json({ success: true, recordId: record.id })
  } catch (error) {
    console.error("Error processing webhook:", error)
    return NextResponse.json({ error: "Failed to process webhook" }, { status: 500 })
  }
}

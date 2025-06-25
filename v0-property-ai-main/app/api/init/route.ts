import { NextResponse } from "next/server"
import { initStorage } from "@/lib/db/image-storage"

// Initialize the image storage when the app starts
initStorage()

export async function GET() {
  return NextResponse.json({ success: true, message: "Image storage initialized" })
}

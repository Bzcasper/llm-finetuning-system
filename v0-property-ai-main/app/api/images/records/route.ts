import { type NextRequest, NextResponse } from "next/server"
import { getAllBlobImageRecords, getUserBlobImageRecords, getBlobImageRecordsByDate } from "@/lib/db/blob-storage"

export async function GET(request: NextRequest) {
  try {
    // Get query parameters
    const userId = request.nextUrl.searchParams.get("userId")
    const date = request.nextUrl.searchParams.get("date")

    let records

    if (userId) {
      // Get records for a specific user
      records = await getUserBlobImageRecords(userId)
    } else if (date) {
      // Get records for a specific date
      records = await getBlobImageRecordsByDate(date)
    } else {
      // Get all records
      records = await getAllBlobImageRecords()
    }

    return NextResponse.json({ success: true, records })
  } catch (error) {
    console.error("Error getting image records:", error)
    return NextResponse.json({ error: "Failed to get image records" }, { status: 500 })
  }
}

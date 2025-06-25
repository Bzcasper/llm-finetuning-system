import { type NextRequest, NextResponse } from "next/server"
import { getServerSession } from "next-auth"
import { authOptions } from "@/lib/auth/config"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function POST(request: NextRequest) {
  try {
    console.log("Onboarding API called")

    const session = await getServerSession(authOptions)
    console.log("Session:", session)

    if (!session?.user?.email) {
      console.log("No session or email found")
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 })
    }

    const data = await request.json()
    console.log("Received data:", data)

    const { fullName, companyName, userType, monthlyVolume, discoverySource } = data

    // Validate required fields
    if (!fullName || !companyName || !userType || !monthlyVolume || !discoverySource) {
      return NextResponse.json({ error: "All fields are required" }, { status: 400 })
    }

    console.log("Updating user profile...")

    // First, check if user exists
    const existingUser = await sql`
      SELECT email FROM users WHERE email = ${session.user.email}
    `

    if (existingUser.length === 0) {
      // Create user if doesn't exist
      await sql`
        INSERT INTO users (email, name, image, created_at, updated_at)
        VALUES (${session.user.email}, ${session.user.name}, ${session.user.image}, NOW(), NOW())
      `
    }

    // Update user profile with onboarding data
    const updateResult = await sql`
      UPDATE users 
      SET 
        name = ${fullName},
        company_name = ${companyName},
        user_type = ${userType},
        monthly_volume = ${monthlyVolume},
        discovery_source = ${discoverySource},
        onboarding_completed = true,
        updated_at = NOW()
      WHERE email = ${session.user.email}
    `

    console.log("User update result:", updateResult)

    // Create user preferences based on their responses
    const creditAllocation = getInitialCredits(monthlyVolume)
    const preferredType = getUserPreferredType(userType)

    console.log("Creating user preferences...")

    await sql`
      INSERT INTO user_preferences (user_email, initial_credits, preferred_enhancement_type, created_at, updated_at)
      VALUES (${session.user.email}, ${creditAllocation}, ${preferredType}, NOW(), NOW())
      ON CONFLICT (user_email) 
      DO UPDATE SET 
        initial_credits = ${creditAllocation},
        preferred_enhancement_type = ${preferredType},
        updated_at = NOW()
    `

    // Log analytics
    await sql`
      INSERT INTO onboarding_analytics (user_email, user_agent, completed_at)
      VALUES (${session.user.email}, ${request.headers.get("user-agent") || "unknown"}, NOW())
    `

    console.log("Onboarding completed successfully")

    return NextResponse.json({
      success: true,
      message: "Onboarding completed successfully",
      credits: creditAllocation,
    })
  } catch (error) {
    console.error("Onboarding error:", error)
    return NextResponse.json(
      {
        error: "Failed to save onboarding data",
        details: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}

function getInitialCredits(monthlyVolume: string): number {
  const volumeMap: Record<string, number> = {
    "1-10": 25,
    "11-50": 100,
    "51-100": 200,
    "101-250": 500,
    "251-500": 1000,
    "500+": 2000,
  }
  return volumeMap[monthlyVolume] || 25
}

function getUserPreferredType(userType: string): string {
  const typeMap: Record<string, string> = {
    "real-estate-agent": "listing_enhancement",
    photographer: "professional_enhancement",
    broker: "batch_processing",
    "property-manager": "quick_enhancement",
    developer: "marketing_materials",
    other: "general_enhancement",
  }
  return typeMap[userType] || "general_enhancement"
}

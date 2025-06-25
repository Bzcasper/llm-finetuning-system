import { type NextRequest, NextResponse } from "next/server"
import { getUserCredits, addCredits, getCreditTransactions } from "@/lib/credit-system"
import { withSecurity } from "@/lib/auth/middleware"
import { Permission } from "@/lib/rbac"

export const GET = withSecurity(
  async (request: NextRequest) => {
    try {
      // Get user ID from the authenticated session
      const userId = request.headers.get("x-user-id") || "demo-user"

      // Get user's current credits
      const credits = await getUserCredits(userId)

      // Get transaction history
      const transactions = await getCreditTransactions(userId)

      return NextResponse.json({
        success: true,
        credits,
        transactions,
        timestamp: new Date().toISOString(),
      })
    } catch (error) {
      console.error("Error getting user credits:", error)

      return NextResponse.json(
        {
          error: "Failed to get user credits",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 },
      )
    }
  },
  {
    requireAuth: true,
    requirePermissions: [Permission.VIEW_OWN_CREDITS],
    rateLimitConfig: "credits",
  }
)

export const POST = withSecurity(
  async (request: NextRequest) => {
    try {
      const body = await request.json()

      // Get user ID from the authenticated session
      const userId = request.headers.get("x-user-id") || "demo-user"

      // Get amount and description
      const { amount, description } = body

      // Validate input
      if (typeof amount !== "number" || amount <= 0) {
        return NextResponse.json(
          { 
            error: "Invalid amount", 
            details: "Amount must be a positive number",
            timestamp: new Date().toISOString(),
          }, 
          { status: 400 }
        )
      }

      if (amount > 1000) {
        return NextResponse.json(
          { 
            error: "Amount too large", 
            details: "Maximum 1000 credits per transaction",
            timestamp: new Date().toISOString(),
          }, 
          { status: 400 }
        )
      }

      // Add credits to user's account
      const success = await addCredits(userId, amount, description || "Added credits")

      if (!success) {
        return NextResponse.json(
          { 
            error: "Failed to add credits",
            timestamp: new Date().toISOString(),
          }, 
          { status: 500 }
        )
      }

      // Get updated credits
      const credits = await getUserCredits(userId)

      return NextResponse.json({
        success: true,
        credits,
        added: amount,
        description: description || "Added credits",
        timestamp: new Date().toISOString(),
      })
    } catch (error) {
      console.error("Error adding credits:", error)

      return NextResponse.json(
        {
          error: "Failed to add credits",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 },
      )
    }
  },
  {
    requireAuth: true,
    requirePermissions: [Permission.USE_CREDITS],
    rateLimitConfig: "credits",
  }
)

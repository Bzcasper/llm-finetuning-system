/**
 * Security middleware test endpoint
 * Used for testing various security scenarios
 */

import { type NextRequest, NextResponse } from "next/server"
import { withSecurity, getUserInfo } from "@/lib/auth/middleware"
import { Permission, UserRole } from "@/lib/rbac"
import { getUserCredits } from "@/lib/credit-system"

// Test endpoint for authenticated users
export const GET = withSecurity(
  async (request: NextRequest) => {
    try {
      const userId = request.headers.get("x-user-id")
      const userEmail = request.headers.get("x-user-email")
      const userRole = request.headers.get("x-user-role") as UserRole

      // Get user credits
      const credits = userId ? await getUserCredits(userId) : 0

      return NextResponse.json({
        success: true,
        message: "Security test passed - user authenticated",
        user: {
          id: userId,
          email: userEmail,
          role: userRole,
          credits,
        },
        timestamp: new Date().toISOString(),
        endpoint: "GET /api/test-security",
      })
    } catch (error) {
      return NextResponse.json(
        {
          error: "Test failed",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      )
    }
  },
  {
    requireAuth: true,
    rateLimitConfig: "api",
  }
)

// Test endpoint for admin users only
export const POST = withSecurity(
  async (request: NextRequest) => {
    try {
      const userId = request.headers.get("x-user-id")
      const userRole = request.headers.get("x-user-role") as UserRole

      return NextResponse.json({
        success: true,
        message: "Admin security test passed",
        user: {
          id: userId,
          role: userRole,
        },
        adminFeatures: [
          "user_management",
          "credit_management",
          "analytics_access",
        ],
        timestamp: new Date().toISOString(),
        endpoint: "POST /api/test-security",
      })
    } catch (error) {
      return NextResponse.json(
        {
          error: "Admin test failed",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      )
    }
  },
  {
    requireAuth: true,
    requireRole: [UserRole.ADMIN, UserRole.SUPER_ADMIN],
    requirePermissions: [Permission.READ_USERS],
    rateLimitConfig: "admin",
  }
)

// Test endpoint for credit checking
export const PUT = withSecurity(
  async (request: NextRequest) => {
    try {
      const userId = request.headers.get("x-user-id")
      const credits = userId ? await getUserCredits(userId) : 0

      return NextResponse.json({
        success: true,
        message: "Credit test passed - user has sufficient credits",
        user: {
          id: userId,
          credits,
        },
        minimumRequired: 5,
        timestamp: new Date().toISOString(),
        endpoint: "PUT /api/test-security",
      })
    } catch (error) {
      return NextResponse.json(
        {
          error: "Credit test failed",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      )
    }
  },
  {
    requireAuth: true,
    checkCredits: true,
    minCredits: 5,
    rateLimitConfig: "api",
  }
)

// Test endpoint for rate limiting (very restrictive)
export const DELETE = withSecurity(
  async (request: NextRequest) => {
    try {
      return NextResponse.json({
        success: true,
        message: "Rate limit test passed",
        info: "This endpoint has very restrictive rate limits",
        timestamp: new Date().toISOString(),
        endpoint: "DELETE /api/test-security",
      })
    } catch (error) {
      return NextResponse.json(
        {
          error: "Rate limit test failed",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      )
    }
  },
  {
    requireAuth: true,
    rateLimitConfig: "auth", // Very restrictive: 5 requests per 15 minutes
  }
)
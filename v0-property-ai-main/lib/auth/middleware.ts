/**
 * Security middleware utilities for PropertyGlow API
 * Provides authentication, authorization, and rate limiting for API routes
 */

import { NextRequest, NextResponse } from "next/server"
import { getToken } from "next-auth/jwt"
import { Session } from "next-auth"
import { 
  rateLimit, 
  RATE_LIMIT_CONFIGS, 
  createRateLimitHeaders, 
  RateLimitResult 
} from "@/lib/rate-limiting"
import { 
  canAccessRoute, 
  getUserRole, 
  UserRole, 
  Permission, 
  createRBACMiddleware,
  canAccessUserResource
} from "@/lib/rbac"
import { getUserCredits, useCredits } from "@/lib/credit-system"

export interface SecurityMiddlewareOptions {
  requireAuth?: boolean
  requireRole?: UserRole | UserRole[]
  requirePermissions?: Permission[]
  rateLimitConfig?: keyof typeof RATE_LIMIT_CONFIGS
  checkCredits?: boolean
  minCredits?: number
  skipRateLimit?: boolean
}

export interface SecurityResult {
  success: boolean
  error?: string
  statusCode?: number
  session?: Session | null
  rateLimitResult?: RateLimitResult
  userRole?: UserRole
  headers?: Record<string, string>
}

/**
 * Main security middleware function
 */
export async function securityMiddleware(
  request: NextRequest,
  options: SecurityMiddlewareOptions = {}
): Promise<SecurityResult> {
  const {
    requireAuth = true,
    requireRole,
    requirePermissions = [],
    rateLimitConfig = "api",
    checkCredits = false,
    minCredits = 1,
    skipRateLimit = false,
  } = options

  try {
    // 1. Rate limiting (unless skipped)
    let rateLimitResult: RateLimitResult | undefined
    let rateLimitHeaders: Record<string, string> = {}

    if (!skipRateLimit) {
      const endpoint = new URL(request.url).pathname
      rateLimitResult = rateLimit(
        request,
        RATE_LIMIT_CONFIGS[rateLimitConfig],
        endpoint
      )

      rateLimitHeaders = createRateLimitHeaders(rateLimitResult)

      if (!rateLimitResult.success) {
        return {
          success: false,
          error: "Rate limit exceeded",
          statusCode: 429,
          rateLimitResult,
          headers: {
            ...rateLimitHeaders,
            "Content-Type": "application/json",
          },
        }
      }
    }

    // 2. Authentication check
    let session: Session | null = null
    
    if (requireAuth) {
      const token = await getToken({ 
        req: request,
        secret: process.env.NEXTAUTH_SECRET 
      })

      if (!token) {
        return {
          success: false,
          error: "Authentication required",
          statusCode: 401,
          headers: {
            ...rateLimitHeaders,
            "Content-Type": "application/json",
          },
        }
      }

      // Create session object from token
      session = {
        user: {
          id: token.sub || token.id || "",
          name: token.name || null,
          email: token.email || null,
          image: token.picture || null,
        },
        expires: new Date(token.exp! * 1000).toISOString(),
      }
    }

    const userRole = session ? getUserRole(session) : UserRole.USER

    // 3. Role-based authorization
    if (requireRole) {
      const allowedRoles = Array.isArray(requireRole) ? requireRole : [requireRole]
      
      if (!allowedRoles.includes(userRole)) {
        return {
          success: false,
          error: "Insufficient role privileges",
          statusCode: 403,
          session,
          userRole,
          headers: {
            ...rateLimitHeaders,
            "Content-Type": "application/json",
          },
        }
      }
    }

    // 4. Permission-based authorization
    if (requirePermissions.length > 0) {
      const rbacCheck = createRBACMiddleware(requirePermissions)(session)
      
      if (!rbacCheck.success) {
        return {
          success: false,
          error: rbacCheck.error,
          statusCode: rbacCheck.statusCode,
          session,
          userRole,
          headers: {
            ...rateLimitHeaders,
            "Content-Type": "application/json",
          },
        }
      }
    }

    // 5. Route-specific authorization
    if (session) {
      const routePath = new URL(request.url).pathname
      const canAccess = canAccessRoute(session, routePath)
      
      if (!canAccess) {
        return {
          success: false,
          error: "Access denied for this route",
          statusCode: 403,
          session,
          userRole,
          headers: {
            ...rateLimitHeaders,
            "Content-Type": "application/json",
          },
        }
      }
    }

    // 6. Credits check (if required)
    if (checkCredits && session) {
      const userCredits = await getUserCredits(session.user.id)
      
      if (userCredits < minCredits) {
        return {
          success: false,
          error: "Insufficient credits",
          statusCode: 402, // Payment Required
          session,
          userRole,
          headers: {
            ...rateLimitHeaders,
            "Content-Type": "application/json",
          },
        }
      }
    }

    // All checks passed
    return {
      success: true,
      session,
      rateLimitResult,
      userRole,
      headers: rateLimitHeaders,
    }

  } catch (error) {
    console.error("Security middleware error:", error)
    
    return {
      success: false,
      error: "Internal security error",
      statusCode: 500,
      headers: {
        "Content-Type": "application/json",
      },
    }
  }
}

/**
 * Create a standardized error response
 */
export function createErrorResponse(
  message: string,
  statusCode: number,
  headers: Record<string, string> = {}
): NextResponse {
  const response = NextResponse.json(
    {
      error: message,
      statusCode,
      timestamp: new Date().toISOString(),
    },
    { status: statusCode }
  )

  // Add security headers
  Object.entries(headers).forEach(([key, value]) => {
    response.headers.set(key, value)
  })

  // Add standard security headers
  response.headers.set("X-Content-Type-Options", "nosniff")
  response.headers.set("X-Frame-Options", "DENY")
  response.headers.set("X-XSS-Protection", "1; mode=block")
  response.headers.set("Referrer-Policy", "strict-origin-when-cross-origin")

  return response
}

/**
 * Wrap an API handler with security middleware
 */
export function withSecurity(
  handler: (request: NextRequest, context?: any) => Promise<NextResponse>,
  options: SecurityMiddlewareOptions = {}
) {
  return async (request: NextRequest, context?: any): Promise<NextResponse> => {
    // Apply security middleware
    const securityResult = await securityMiddleware(request, options)

    if (!securityResult.success) {
      return createErrorResponse(
        securityResult.error || "Security check failed",
        securityResult.statusCode || 500,
        securityResult.headers
      )
    }

    try {
      // Add user information to request headers for handler access
      if (securityResult.session) {
        request.headers.set("x-user-id", securityResult.session.user.id)
        request.headers.set("x-user-email", securityResult.session.user.email || "")
        request.headers.set("x-user-role", securityResult.userRole || UserRole.USER)
      }

      // Call the original handler
      const response = await handler(request, context)

      // Add rate limit headers to successful responses
      if (securityResult.headers) {
        Object.entries(securityResult.headers).forEach(([key, value]) => {
          response.headers.set(key, value)
        })
      }

      return response

    } catch (error) {
      console.error("API handler error:", error)
      
      return createErrorResponse(
        "Internal server error",
        500,
        securityResult.headers
      )
    }
  }
}

/**
 * Helper function to check user resource ownership
 */
export async function checkResourceOwnership(
  request: NextRequest,
  resourceUserId: string
): Promise<SecurityResult> {
  const token = await getToken({ 
    req: request,
    secret: process.env.NEXTAUTH_SECRET 
  })

  if (!token) {
    return {
      success: false,
      error: "Authentication required",
      statusCode: 401,
    }
  }

  const session: Session = {
    user: {
      id: token.sub || token.id || "",
      name: token.name || null,
      email: token.email || null,
      image: token.picture || null,
    },
    expires: new Date(token.exp! * 1000).toISOString(),
  }

  const canAccess = canAccessUserResource(session, resourceUserId)

  if (!canAccess) {
    return {
      success: false,
      error: "Access denied - insufficient permissions for this resource",
      statusCode: 403,
      session,
    }
  }

  return {
    success: true,
    session,
  }
}

/**
 * Middleware specifically for credit-consuming operations
 */
export async function withCreditCheck(
  handler: (request: NextRequest, session: Session) => Promise<NextResponse>,
  creditCost: number = 1
) {
  return async (request: NextRequest): Promise<NextResponse> => {
    // Apply security with credit checking
    const securityResult = await securityMiddleware(request, {
      requireAuth: true,
      checkCredits: true,
      minCredits: creditCost,
      rateLimitConfig: "enhance",
    })

    if (!securityResult.success || !securityResult.session) {
      return createErrorResponse(
        securityResult.error || "Security check failed",
        securityResult.statusCode || 500,
        securityResult.headers
      )
    }

    try {
      // Deduct credits before processing
      const creditUsed = await useCredits(
        securityResult.session.user.id,
        creditCost,
        `API usage: ${new URL(request.url).pathname}`
      )

      if (!creditUsed) {
        return createErrorResponse(
          "Failed to deduct credits",
          402,
          securityResult.headers
        )
      }

      // Call the original handler
      const response = await handler(request, securityResult.session)

      // Add rate limit headers
      if (securityResult.headers) {
        Object.entries(securityResult.headers).forEach(([key, value]) => {
          response.headers.set(key, value)
        })
      }

      return response

    } catch (error) {
      console.error("Credit-checked API handler error:", error)
      
      // On error, try to refund credits (in a real app, use a transaction)
      try {
        await useCredits(
          securityResult.session.user.id,
          -creditCost,
          `Refund for failed API usage: ${new URL(request.url).pathname}`
        )
      } catch (refundError) {
        console.error("Failed to refund credits:", refundError)
      }
      
      return createErrorResponse(
        "Internal server error",
        500,
        securityResult.headers
      )
    }
  }
}

/**
 * Extract user information from request
 */
export async function getUserInfo(request: NextRequest): Promise<{
  userId?: string
  session?: Session
  userRole?: UserRole
}> {
  try {
    const token = await getToken({ 
      req: request,
      secret: process.env.NEXTAUTH_SECRET 
    })

    if (!token) {
      return {}
    }

    const session: Session = {
      user: {
        id: token.sub || token.id || "",
        name: token.name || null,
        email: token.email || null,
        image: token.picture || null,
      },
      expires: new Date(token.exp! * 1000).toISOString(),
    }

    return {
      userId: session.user.id,
      session,
      userRole: getUserRole(session),
    }
  } catch (error) {
    console.error("Error extracting user info:", error)
    return {}
  }
}
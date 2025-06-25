/**
 * Rate limiting utilities for PropertyGlow API
 * Implements memory-based rate limiting with configurable limits
 */

import { NextRequest } from "next/server"

export interface RateLimitConfig {
  windowMs: number // Time window in milliseconds
  maxRequests: number // Maximum requests per window
  skipSuccessfulRequests?: boolean // Whether to skip counting successful requests
  skipFailedRequests?: boolean // Whether to skip counting failed requests
}

export interface RateLimitResult {
  success: boolean
  limit: number
  remaining: number
  reset: Date
  retryAfter?: number
}

// In-memory storage for rate limiting
// In production, use Redis or similar distributed cache
const rateLimitStore = new Map<string, { count: number; resetTime: number }>()

// Default rate limit configurations
export const RATE_LIMIT_CONFIGS = {
  default: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    maxRequests: 100, // 100 requests per 15 minutes
  },
  api: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 100, // 100 requests per minute per IP
  },
  auth: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    maxRequests: 5, // 5 auth attempts per 15 minutes
  },
  enhance: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 10, // 10 enhancement requests per minute
  },
  credits: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 50, // 50 requests per minute
  },
  stripe: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 20, // 20 requests per minute
  },
  admin: {
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 200, // 200 requests per minute for admin operations
  },
} as const

/**
 * Get client IP address from request
 */
export function getClientIP(request: NextRequest): string {
  // Check for IP in various headers (proxy-aware)
  const forwarded = request.headers.get("x-forwarded-for")
  const realIP = request.headers.get("x-real-ip")
  const clientIP = request.headers.get("x-client-ip")
  
  if (forwarded) {
    // x-forwarded-for can contain multiple IPs, take the first one
    return forwarded.split(",")[0].trim()
  }
  
  if (realIP) {
    return realIP
  }
  
  if (clientIP) {
    return clientIP
  }
  
  // Fallback to connection IP
  return request.ip || "unknown"
}

/**
 * Generate rate limit key
 */
export function getRateLimitKey(
  identifier: string,
  endpoint: string,
  userId?: string
): string {
  // Include user ID if available for per-user limits
  const userPart = userId ? `:${userId}` : ""
  return `rate_limit:${endpoint}:${identifier}${userPart}`
}

/**
 * Check rate limit for a given key
 */
export function checkRateLimit(
  key: string,
  config: RateLimitConfig
): RateLimitResult {
  const now = Date.now()
  const windowStart = now - config.windowMs
  
  // Clean up expired entries
  for (const [k, v] of rateLimitStore.entries()) {
    if (v.resetTime < now) {
      rateLimitStore.delete(k)
    }
  }
  
  const current = rateLimitStore.get(key)
  
  if (!current || current.resetTime < now) {
    // First request in window or expired window
    const resetTime = now + config.windowMs
    rateLimitStore.set(key, { count: 1, resetTime })
    
    return {
      success: true,
      limit: config.maxRequests,
      remaining: config.maxRequests - 1,
      reset: new Date(resetTime),
    }
  }
  
  if (current.count >= config.maxRequests) {
    // Rate limit exceeded
    const retryAfter = Math.ceil((current.resetTime - now) / 1000)
    
    return {
      success: false,
      limit: config.maxRequests,
      remaining: 0,
      reset: new Date(current.resetTime),
      retryAfter,
    }
  }
  
  // Increment counter
  current.count++
  rateLimitStore.set(key, current)
  
  return {
    success: true,
    limit: config.maxRequests,
    remaining: config.maxRequests - current.count,
    reset: new Date(current.resetTime),
  }
}

/**
 * Apply rate limiting to a request
 */
export function rateLimit(
  request: NextRequest,
  config: RateLimitConfig,
  endpoint: string,
  userId?: string
): RateLimitResult {
  const clientIP = getClientIP(request)
  const key = getRateLimitKey(clientIP, endpoint, userId)
  
  return checkRateLimit(key, config)
}

/**
 * Create rate limit headers for response
 */
export function createRateLimitHeaders(result: RateLimitResult): Record<string, string> {
  const headers: Record<string, string> = {
    "X-RateLimit-Limit": result.limit.toString(),
    "X-RateLimit-Remaining": result.remaining.toString(),
    "X-RateLimit-Reset": result.reset.getTime().toString(),
  }
  
  if (result.retryAfter) {
    headers["Retry-After"] = result.retryAfter.toString()
  }
  
  return headers
}

/**
 * Middleware wrapper for rate limiting
 */
export function createRateLimitMiddleware(
  config: RateLimitConfig,
  endpoint: string
) {
  return (request: NextRequest, userId?: string) => {
    const result = rateLimit(request, config, endpoint, userId)
    
    return {
      ...result,
      headers: createRateLimitHeaders(result),
    }
  }
}

/**
 * Clean up expired rate limit entries (should be called periodically)
 */
export function cleanupRateLimitStore(): void {
  const now = Date.now()
  
  for (const [key, value] of rateLimitStore.entries()) {
    if (value.resetTime < now) {
      rateLimitStore.delete(key)
    }
  }
}

// Clean up expired entries every 5 minutes
if (typeof window === "undefined") {
  setInterval(cleanupRateLimitStore, 5 * 60 * 1000)
}
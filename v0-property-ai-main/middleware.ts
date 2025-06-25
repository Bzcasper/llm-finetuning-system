import { withAuth } from "next-auth/middleware"
import { NextResponse } from "next/server"
import { 
  rateLimit, 
  RATE_LIMIT_CONFIGS, 
  createRateLimitHeaders 
} from "@/lib/rate-limiting"
import { 
  canAccessRoute, 
  getUserRole, 
  UserRole 
} from "@/lib/rbac"
import { Session } from "next-auth"

export default withAuth(
  async function middleware(req) {
    const { pathname } = req.nextUrl
    
    // Allow access to auth pages, static assets, and public pages
    if (
      pathname.startsWith('/auth') ||
      pathname.startsWith('/api/auth') ||
      pathname.startsWith('/_next') ||
      pathname.startsWith('/favicon.ico') ||
      pathname === '/' ||
      pathname.startsWith('/api/env-check') // Allow env check for debugging
    ) {
      return NextResponse.next()
    }

    // Apply rate limiting to API routes
    if (pathname.startsWith('/api/')) {
      const rateLimitConfig = getRateLimitConfigForRoute(pathname)
      const rateLimitResult = rateLimit(req, rateLimitConfig, pathname)
      
      if (!rateLimitResult.success) {
        const headers = createRateLimitHeaders(rateLimitResult)
        return new NextResponse(
          JSON.stringify({
            error: "Rate limit exceeded",
            statusCode: 429,
            retryAfter: rateLimitResult.retryAfter,
          }),
          {
            status: 429,
            headers: {
              ...headers,
              "Content-Type": "application/json",
            },
          }
        )
      }
    }

    // Protect all other routes - require authentication
    if (!req.nextauth.token) {
      if (pathname.startsWith('/api/')) {
        // Return 401 for API routes
        return new NextResponse(
          JSON.stringify({
            error: "Authentication required",
            statusCode: 401,
          }),
          {
            status: 401,
            headers: {
              "Content-Type": "application/json",
            },
          }
        )
      }
      
      // Redirect to sign-in for UI routes
      const signInUrl = new URL('/auth/signin', req.url)
      signInUrl.searchParams.set('callbackUrl', req.url)
      return NextResponse.redirect(signInUrl)
    }

    // Additional checks for API routes
    if (pathname.startsWith('/api/')) {
      const session: Session = {
        user: {
          id: req.nextauth.token.sub || req.nextauth.token.id || "",
          name: req.nextauth.token.name || null,
          email: req.nextauth.token.email || null,
          image: req.nextauth.token.picture || null,
        },
        expires: new Date(req.nextauth.token.exp! * 1000).toISOString(),
      }

      // Check route-specific permissions
      const canAccess = canAccessRoute(session, pathname)
      
      if (!canAccess) {
        return new NextResponse(
          JSON.stringify({
            error: "Access denied - insufficient permissions",
            statusCode: 403,
          }),
          {
            status: 403,
            headers: {
              "Content-Type": "application/json",
            },
          }
        )
      }

      // Special handling for admin routes
      if (pathname.startsWith('/api/admin/')) {
        const userRole = getUserRole(session)
        
        if (userRole !== UserRole.ADMIN && userRole !== UserRole.SUPER_ADMIN) {
          return new NextResponse(
            JSON.stringify({
              error: "Admin access required",
              statusCode: 403,
            }),
            {
              status: 403,
              headers: {
                "Content-Type": "application/json",
              },
            }
          )
        }
      }
    }

    return NextResponse.next()
  },
  {
    callbacks: {
      authorized: ({ token, req }) => {
        const { pathname } = req.nextUrl
        
        // Always allow access to auth pages, home page, and public API routes
        if (
          pathname.startsWith('/auth') ||
          pathname.startsWith('/api/auth') ||
          pathname === '/' ||
          pathname.startsWith('/api/env-check')
        ) {
          return true
        }

        // For protected routes, require a token
        return !!token
      },
    },
  }
)

/**
 * Get appropriate rate limit configuration for a route
 */
function getRateLimitConfigForRoute(pathname: string) {
  if (pathname.startsWith('/api/enhance')) {
    return RATE_LIMIT_CONFIGS.enhance
  }
  
  if (pathname.startsWith('/api/credits')) {
    return RATE_LIMIT_CONFIGS.credits
  }
  
  if (pathname.startsWith('/api/stripe')) {
    return RATE_LIMIT_CONFIGS.stripe
  }
  
  if (pathname.startsWith('/api/admin')) {
    return RATE_LIMIT_CONFIGS.admin
  }
  
  if (pathname.startsWith('/api/auth')) {
    return RATE_LIMIT_CONFIGS.auth
  }
  
  // Default API rate limit
  return RATE_LIMIT_CONFIGS.api
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * Now includes API routes for security middleware
     */
    '/((?!_next/static|_next/image|favicon.ico).*)',
  ],
}
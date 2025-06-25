// Ensure this route runs in the Node.js runtime (NOT Edge) and is always dynamic
export const runtime = "nodejs"
export const dynamic = "force-dynamic"

import NextAuth from "next-auth"
import { authOptions } from "@/lib/auth/config"
import { NextRequest, NextResponse } from "next/server"

// Create the NextAuth handler
const handler = NextAuth(authOptions)

// Add error handling wrapper
async function authHandler(req: NextRequest, context: { params: { nextauth: string[] } }) {
  try {
    return await handler(req, context)
  } catch (error) {
    console.error("NextAuth API Error:", error)
    
    // Return a proper error response
    return NextResponse.json(
      { 
        error: "Authentication service temporarily unavailable",
        message: process.env.NODE_ENV === "development" ? error.toString() : "Please try again later"
      },
      { status: 500 }
    )
  }
}

export { authHandler as GET, authHandler as POST }

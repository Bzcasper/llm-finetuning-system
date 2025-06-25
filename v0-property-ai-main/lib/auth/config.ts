import type { NextAuthOptions } from "next-auth"
import GoogleProvider from "next-auth/providers/google"
import CredentialsProvider from "next-auth/providers/credentials"
import { NeonAdapter } from "./neon-adapter"
import { neon } from "@neondatabase/serverless"

// Environment validation with better error messages
const requiredEnvVars = {
  NEXTAUTH_SECRET: process.env.NEXTAUTH_SECRET,
  DATABASE_URL: process.env.DATABASE_URL,
  GOOGLE_CLIENT_ID: process.env.GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET: process.env.GOOGLE_CLIENT_SECRET,
}

for (const [key, value] of Object.entries(requiredEnvVars)) {
  if (!value) {
    throw new Error(`Missing required environment variable: ${key}. Please add it to your .env.local file.`)
  }
}

// Determine base URL with better fallback handling
const getBaseUrl = (): string => {
  if (process.env.NEXTAUTH_URL) {
    return process.env.NEXTAUTH_URL
  }
  
  if (process.env.VERCEL_URL) {
    return `https://${process.env.VERCEL_URL}`
  }
  
  if (process.env.NODE_ENV === "development") {
    return "http://localhost:3000"
  }
  
  throw new Error("NEXTAUTH_URL must be set in production")
}

const NEXTAUTH_URL = getBaseUrl()

// Create database connection with error handling
const sql = neon(process.env.DATABASE_URL!)

export const authOptions: NextAuthOptions = {
  debug: process.env.NODE_ENV === "development",
  adapter: NeonAdapter(sql),
  secret: process.env.NEXTAUTH_SECRET,
  trustHost: true,
  // Fix for CLIENT_FETCH_ERROR in Next.js 15
  useSecureCookies: process.env.NODE_ENV === "production",
  cookies: {
    sessionToken: {
      name: process.env.NODE_ENV === "production" ? "__Secure-next-auth.session-token" : "next-auth.session-token",
      options: {
        httpOnly: true,
        sameSite: "lax",
        path: "/",
        secure: process.env.NODE_ENV === "production",
      },
    },
  },
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    CredentialsProvider({
      name: "credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null
        }

        // For demo purposes only - in production, use proper password hashing
        if (credentials.email === "demo@example.com" && credentials.password === "password123") {
          return {
            id: "1",
            name: "Demo User",
            email: "demo@example.com",
            image: "/placeholder-user.jpg",
          }
        }

        return null
      },
    }),
  ],
  callbacks: {
    async session({ session, token, user }) {
      // Include user ID in session for both JWT and database sessions
      if (session.user) {
        if (token?.sub) {
          session.user.id = token.sub
        } else if (user?.id) {
          session.user.id = user.id
        }
      }
      return session
    },
    async jwt({ token, user, account }) {
      // Persist user data to the token
      if (user) {
        token.id = user.id
      }
      // Persist OAuth account data
      if (account) {
        token.accessToken = account.access_token
        token.refreshToken = account.refresh_token
      }
      return token
    },
    async signIn({ user, account, profile }) {
      // Allow sign in for all users
      return true
    },
    async redirect({ url, baseUrl }) {
      // Allows relative callback URLs
      if (url.startsWith("/")) return `${baseUrl}${url}`
      // Allows callback URLs on the same origin
      else if (new URL(url).origin === baseUrl) return url
      return baseUrl
    },
  },
  pages: {
    signIn: "/auth/signin",
    signUp: "/auth/signup",
    signOut: "/auth/signout",
    error: "/auth/error",
  },
  session: {
    strategy: "database", // Use database sessions for better security and reliability
    maxAge: 30 * 24 * 60 * 60, // 30 days
    updateAge: 24 * 60 * 60, // 24 hours
  },
  // Add error handling
  events: {
    async error(message) {
      console.error("NextAuth Error:", message)
    },
  },
}

export const APP_BASE_URL = NEXTAUTH_URL

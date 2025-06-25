import { DefaultSession, DefaultUser } from "next-auth"
import { JWT, DefaultJWT } from "next-auth/jwt"

declare module "next-auth" {
  interface Session {
    user: {
      id: string
      credits?: number
      subscriptionTier?: string
      subscriptionStatus?: string
    } & DefaultSession["user"]
  }

  interface User extends DefaultUser {
    id: string
    credits?: number
    subscriptionTier?: string
    subscriptionStatus?: string
  }
}

declare module "next-auth/jwt" {
  interface JWT extends DefaultJWT {
    id: string
    accessToken?: string
    refreshToken?: string
    credits?: number
    subscriptionTier?: string
    subscriptionStatus?: string
  }
}
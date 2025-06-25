import { getServerSession } from "next-auth/next"
import { authOptions } from "./config"
import { redirect } from "next/navigation"

export async function getSession() {
  return await getServerSession(authOptions)
}

export async function getCurrentUser() {
  const session = await getSession()
  return session?.user
}

export async function requireAuth() {
  const session = await getSession()
  
  if (!session) {
    redirect("/auth/signin")
  }
  
  return session
}

export async function requireUser() {
  const session = await requireAuth()
  return session.user
}
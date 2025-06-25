// Credit system for PropertyGlow

import { cookies } from "next/headers"

// In a real app, this would be stored in a database
// For this demo, we'll use cookies to simulate persistence
const CREDIT_COOKIE_NAME = "property-glow-credits"

export type CreditTransaction = {
  id: string
  userId: string
  amount: number // Positive for additions, negative for usage
  description: string
  timestamp: number
}

export async function getUserCredits(userId: string): Promise<number> {
  // In a real app, this would fetch from a database
  // For this demo, we'll use cookies
  const cookieStore = cookies()
  const creditCookie = cookieStore.get(CREDIT_COOKIE_NAME)

  if (!creditCookie) {
    // New user, give them $1 worth of credits (10 credits)
    await addCredits(userId, 10, "Welcome bonus")
    return 10
  }

  try {
    const creditData = JSON.parse(creditCookie.value)
    return creditData.credits || 0
  } catch (error) {
    console.error("Error parsing credit cookie:", error)
    return 0
  }
}

export async function addCredits(userId: string, amount: number, description: string): Promise<boolean> {
  // In a real app, this would update a database
  // For this demo, we'll use cookies
  const cookieStore = cookies()
  const creditCookie = cookieStore.get(CREDIT_COOKIE_NAME)

  let creditData = { credits: 0, transactions: [] }

  if (creditCookie) {
    try {
      creditData = JSON.parse(creditCookie.value)
    } catch (error) {
      console.error("Error parsing credit cookie:", error)
    }
  }

  // Add the transaction
  const transaction: CreditTransaction = {
    id: Math.random().toString(36).substring(2, 15),
    userId,
    amount,
    description,
    timestamp: Date.now(),
  }

  creditData.credits += amount
  creditData.transactions = [...(creditData.transactions || []), transaction]

  // Save the updated credit data
  cookieStore.set(CREDIT_COOKIE_NAME, JSON.stringify(creditData), {
    maxAge: 60 * 60 * 24 * 30, // 30 days
    path: "/",
  })

  return true
}

export async function useCredits(userId: string, amount: number, description: string): Promise<boolean> {
  // Check if user has enough credits
  const currentCredits = await getUserCredits(userId)

  if (currentCredits < amount) {
    return false // Not enough credits
  }

  // Use the credits (add negative amount)
  return addCredits(userId, -amount, description)
}

export async function getCreditTransactions(userId: string): Promise<CreditTransaction[]> {
  // In a real app, this would fetch from a database
  // For this demo, we'll use cookies
  const cookieStore = cookies()
  const creditCookie = cookieStore.get(CREDIT_COOKIE_NAME)

  if (!creditCookie) {
    return []
  }

  try {
    const creditData = JSON.parse(creditCookie.value)
    return creditData.transactions || []
  } catch (error) {
    console.error("Error parsing credit cookie:", error)
    return []
  }
}

/**
 * Stripe Webhook Handler
 * Protected with basic authentication and rate limiting
 */

import { type NextRequest, NextResponse } from "next/server"
import { withSecurity } from "@/lib/auth/middleware"
import { addCredits } from "@/lib/credit-system"

export const POST = withSecurity(
  async (request: NextRequest) => {
    try {
      // Verify stripe webhook signature (in production)
      const signature = request.headers.get("stripe-signature")
      
      if (!signature) {
        return NextResponse.json(
          {
            error: "Missing Stripe signature",
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        )
      }

      const body = await request.text()
      
      // In a real app, verify the webhook signature with Stripe
      // const event = stripe.webhooks.constructEvent(body, signature, webhookSecret)
      
      // Mock webhook event processing
      const mockEvent = {
        type: "checkout.session.completed",
        data: {
          object: {
            id: "cs_test_123",
            customer_email: "user@example.com",
            amount_total: 1000, // $10.00 in cents
            metadata: {
              user_id: "user_123",
              credits: "100",
            },
          },
        },
      }

      console.log("Stripe webhook received:", {
        type: mockEvent.type,
        sessionId: mockEvent.data.object.id,
        customerEmail: mockEvent.data.object.customer_email,
      })

      // Process different event types
      switch (mockEvent.type) {
        case "checkout.session.completed":
          const session = mockEvent.data.object
          const userId = session.metadata.user_id
          const credits = parseInt(session.metadata.credits || "0")
          
          if (userId && credits > 0) {
            const success = await addCredits(
              userId,
              credits,
              `Stripe payment - Session ${session.id}`
            )
            
            if (success) {
              console.log(`Added ${credits} credits to user ${userId}`)
            } else {
              console.error(`Failed to add credits to user ${userId}`)
            }
          }
          break

        case "invoice.payment_succeeded":
          // Handle subscription payments
          console.log("Subscription payment succeeded")
          break

        case "invoice.payment_failed":
          // Handle failed payments
          console.log("Payment failed")
          break

        default:
          console.log(`Unhandled event type: ${mockEvent.type}`)
      }

      return NextResponse.json({
        success: true,
        received: true,
        eventType: mockEvent.type,
        timestamp: new Date().toISOString(),
      })
    } catch (error) {
      console.error("Stripe webhook error:", error)

      return NextResponse.json(
        {
          error: "Webhook processing failed",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      )
    }
  },
  {
    requireAuth: false, // Webhooks don't require user authentication
    rateLimitConfig: "stripe",
    skipRateLimit: false, // Still apply rate limiting for webhooks
  }
)

// Create payment session endpoint
export const GET = withSecurity(
  async (request: NextRequest) => {
    try {
      const { searchParams } = new URL(request.url)
      const amount = parseInt(searchParams.get("amount") || "1000") // Default $10
      const credits = parseInt(searchParams.get("credits") || "100") // Default 100 credits

      // Validate input
      if (amount < 100 || amount > 100000) { // $1 to $1000
        return NextResponse.json(
          {
            error: "Invalid amount",
            details: "Amount must be between $1 and $1000",
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        )
      }

      if (credits < 1 || credits > 10000) {
        return NextResponse.json(
          {
            error: "Invalid credits amount",
            details: "Credits must be between 1 and 10000",
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        )
      }

      // In a real app, create Stripe checkout session
      const mockSession = {
        id: "cs_test_" + Math.random().toString(36).substring(2, 15),
        url: `https://checkout.stripe.com/pay/cs_test_${Math.random().toString(36).substring(2, 15)}`,
        amount_total: amount,
        currency: "usd",
        metadata: {
          credits: credits.toString(),
        },
      }

      console.log("Created Stripe checkout session:", {
        sessionId: mockSession.id,
        amount: amount,
        credits: credits,
      })

      return NextResponse.json({
        success: true,
        session: mockSession,
        checkoutUrl: mockSession.url,
        timestamp: new Date().toISOString(),
      })
    } catch (error) {
      console.error("Error creating payment session:", error)

      return NextResponse.json(
        {
          error: "Failed to create payment session",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      )
    }
  },
  {
    requireAuth: true,
    rateLimitConfig: "stripe",
  }
)
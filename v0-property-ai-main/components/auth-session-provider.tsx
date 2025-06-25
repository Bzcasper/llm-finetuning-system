"use client"

import { SessionProvider } from "next-auth/react"
import { type ReactNode, useState, useEffect } from "react"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { AlertCircle, X } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function AuthSessionProvider({ children }: { children: ReactNode }) {
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    // Listen for NextAuth errors and network errors
    const handleError = (event: ErrorEvent | PromiseRejectionEvent) => {
      const errorMessage = event instanceof ErrorEvent ? event.error?.toString() : event.reason?.toString()
      
      if (errorMessage && (
        errorMessage.includes("next-auth") || 
        errorMessage.includes("CLIENT_FETCH_ERROR") ||
        errorMessage.includes("fetch")
      )) {
        console.error("Auth error:", errorMessage)
        setError("Authentication service is experiencing issues. Please refresh the page or try again later.")
      }
    }

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      handleError(event)
    }

    window.addEventListener("error", handleError)
    window.addEventListener("unhandledrejection", handleUnhandledRejection)
    
    return () => {
      window.removeEventListener("error", handleError)
      window.removeEventListener("unhandledrejection", handleUnhandledRejection)
    }
  }, [])

  return (
    <SessionProvider
      // Reduce refetch interval to avoid excessive requests
      refetchInterval={5 * 60} // 5 minutes
      refetchOnWindowFocus={false}
      // Add base path for apps running on subpaths
      basePath="/api/auth"
    >
      {error && (
        <div className="fixed top-4 right-4 z-50 max-w-md">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="flex items-center justify-between">
              <span>{error}</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setError(null)}
                className="ml-2 h-auto p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </AlertDescription>
          </Alert>
        </div>
      )}
      {children}
    </SessionProvider>
  )
}

"use client"

import { useSession } from "next-auth/react"
import { Button } from "@/components/ui/button"
import { signIn, signOut } from "next-auth/react"
import { Loader2 } from "lucide-react"

export function AuthStatus() {
  const { data: session, status } = useSession()

  if (status === "loading") {
    return (
      <div className="flex items-center gap-2">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>Loading...</span>
      </div>
    )
  }

  if (status === "authenticated") {
    return (
      <div className="flex items-center gap-4">
        <span>
          Signed in as <strong>{session.user?.name || session.user?.email}</strong>
        </span>
        <Button variant="outline" size="sm" onClick={() => signOut()}>
          Sign out
        </Button>
      </div>
    )
  }

  return (
    <Button variant="outline" size="sm" onClick={() => signIn()}>
      Sign in
    </Button>
  )
}

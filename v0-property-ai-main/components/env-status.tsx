"use client"

import { useEffect, useState } from "react"
import { AlertCircle, CheckCircle } from "lucide-react"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

export default function EnvStatus() {
  const [status, setStatus] = useState<{
    ready: boolean
    message: string
  } | null>(null)

  useEffect(() => {
    const checkEnvStatus = async () => {
      try {
        const res = await fetch("/api/env-check")
        const data = await res.json()

        setStatus({
          ready: data.ready,
          message: data.message,
        })
      } catch (error) {
        setStatus({
          ready: false,
          message: "Failed to check environment status",
        })
      }
    }

    checkEnvStatus()
  }, [])

  if (!status) return null

  return (
    <Alert variant={status.ready ? "default" : "destructive"}>
      {status.ready ? <CheckCircle className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
      <AlertTitle>{status.ready ? "System Ready" : "Configuration Issue"}</AlertTitle>
      <AlertDescription>{status.message}</AlertDescription>
    </Alert>
  )
}

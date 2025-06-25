"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Loader2 } from "lucide-react"

export default function ExamplePage() {
  const [result, setResult] = useState<string | null>(null)
  const [requestId, setRequestId] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [logs, setLogs] = useState<string[]>([])

  const addLog = (message: string) => {
    setLogs((prev) => [...prev, `[${new Date().toISOString()}] ${message}`])
  }

  const runExample = async () => {
    setIsProcessing(true)
    setError(null)
    setResult(null)
    setRequestId(null)
    addLog("Starting example code execution")

    try {
      const response = await fetch("/api/example", {
        method: "POST",
      })

      addLog(`Received response: ${response.status} ${response.statusText}`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to process example")
      }

      if (!data.imageBase64) {
        throw new Error("No image was returned")
      }

      addLog(`Successfully received generated image (Request ID: ${data.requestId})`)
      setResult(data.imageBase64)
      setRequestId(data.requestId)
    } catch (err) {
      console.error("Error running example:", err)
      addLog(`Error: ${err instanceof Error ? err.message : "Unknown error"}`)
      setError(err instanceof Error ? err.message : "An unknown error occurred")
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="container max-w-4xl py-8">
      <h1 className="text-2xl font-bold mb-4">Official Example Code Test</h1>
      <p className="text-gray-500 mb-6">This test runs the example code from the official Fal AI documentation.</p>

      <div className="space-y-6">
        <div className="border p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-2">Run Example Code</h2>
          <p className="text-sm text-gray-500 mb-4">
            This will run the example code that generates an image of "a cat" using the flux/dev model.
          </p>
          <Button onClick={runExample} disabled={isProcessing} className="bg-rose-500 hover:bg-rose-600">
            {isProcessing ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Running Example...
              </>
            ) : (
              "Run Example"
            )}
          </Button>

          {error && <div className="mt-4 p-3 bg-red-50 text-red-700 rounded-md">{error}</div>}
        </div>

        {result && (
          <div className="border p-4 rounded-lg">
            <h2 className="text-lg font-medium mb-2">Result</h2>
            {requestId && <p className="text-xs text-gray-500 mb-2">Request ID: {requestId}</p>}
            <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
              <img src={result || "/placeholder.svg"} alt="Generated" className="w-full h-full object-contain" />
            </div>
          </div>
        )}

        <div className="border p-4 rounded-lg">
          <h2 className="text-lg font-medium mb-2">Logs</h2>
          <div className="bg-gray-900 text-gray-100 p-4 rounded-md font-mono text-xs h-64 overflow-y-auto">
            {logs.length > 0 ? (
              logs.map((log, i) => <div key={i}>{log}</div>)
            ) : (
              <div className="text-gray-500">No logs yet</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

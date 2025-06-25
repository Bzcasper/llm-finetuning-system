import Link from "next/link"
import { Button } from "@/components/ui/button"
import { ArrowLeft } from "lucide-react"
import ImageProcessor from "@/components/image-processor"
import EnvStatus from "@/components/env-status"

export default function TestPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b bg-white">
        <div className="container flex h-16 items-center justify-between py-4">
          <div className="flex items-center gap-2">
            <Link href="/dashboard">
              <Button variant="ghost" size="sm" className="gap-2">
                <ArrowLeft className="h-4 w-4" />
                Back to Dashboard
              </Button>
            </Link>
          </div>
        </div>
      </header>

      <main className="py-8">
        <div className="container max-w-4xl">
          <div className="mb-8">
            <h1 className="text-3xl font-bold">Fal AI Integration Test</h1>
            <p className="text-gray-500 mt-1">
              Upload an image to test the Fal AI integration and verify that image enhancement is working.
            </p>
          </div>

          <div className="space-y-6">
            <EnvStatus />
            <ImageProcessor />
          </div>
        </div>
      </main>
    </div>
  )
}

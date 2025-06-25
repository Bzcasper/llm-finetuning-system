import type { Metadata } from "next"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import ImageUploader from "@/components/image-uploader"
import EnhancementOptions from "@/components/enhancement-options"
import ProcessedGallery from "@/components/processed-gallery"
// Add import for EnvStatus
import EnvStatus from "@/components/env-status"

export const metadata: Metadata = {
  title: "Dashboard - PropertyGlow",
  description: "Enhance your real estate photos with AI",
}

export default function DashboardPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="border-b">
        <div className="container flex h-16 items-center justify-between py-4">
          <div className="flex items-center gap-2">
            <Link href="/" className="flex items-center gap-2">
              <span className="text-xl font-bold">PropertyGlow</span>
            </Link>
          </div>
          <nav className="hidden md:flex items-center gap-6">
            <Link href="/dashboard" className="text-sm font-medium text-rose-500">
              Dashboard
            </Link>
            <Link href="/dashboard/history" className="text-sm font-medium">
              History
            </Link>
            <Link href="/dashboard/settings" className="text-sm font-medium">
              Settings
            </Link>
          </nav>
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="sm">
              Help
            </Button>
            <Button variant="outline" size="sm">
              Account
            </Button>
          </div>
        </div>
      </header>
      <main className="flex-1 py-6">
        <div className="container">
          <div className="flex flex-col gap-8">
            <div>
              <h1 className="text-3xl font-bold">Dashboard</h1>
              <p className="text-gray-500">Enhance your property photos with AI</p>
            </div>

            {/* Add EnvStatus component */}
            <EnvStatus />

            {/* Add Test Integration buttons */}
            <div className="flex flex-wrap gap-4">
              <Link href="/pro-enhance">
                <Button className="bg-rose-500 hover:bg-rose-600">Professional Enhancement Suite</Button>
              </Link>
              <Link href="/simple-enhance">
                <Button className="bg-green-600 hover:bg-green-700 text-white">Simple Tiered Enhancement</Button>
              </Link>
              <Link href="/model-test">
                <Button variant="outline">Real Estate Model Test</Button>
              </Link>
              <Link href="/image-history">
                <Button className="bg-blue-600 hover:bg-blue-700 text-white">Image History</Button>
              </Link>
              <Link href="/test">
                <Button variant="outline">Test Fal AI Integration</Button>
              </Link>
              <Link href="/simple-test">
                <Button variant="outline">Simple Test (Debug Mode)</Button>
              </Link>
              <Link href="/direct-test">
                <Button variant="outline">Direct API Test (Bypass AI SDK)</Button>
              </Link>
              <Link href="/fal-example">
                <Button variant="outline">Fal.ai Example</Button>
              </Link>
            </div>

            <Tabs defaultValue="upload" className="w-full">
              <TabsList className="grid w-full max-w-md grid-cols-3">
                <TabsTrigger value="upload">Upload</TabsTrigger>
                <TabsTrigger value="enhance">Enhance</TabsTrigger>
                <TabsTrigger value="results">Results</TabsTrigger>
              </TabsList>
              <TabsContent value="upload" className="mt-6">
                <div className="grid gap-6">
                  <div className="rounded-lg border p-6">
                    <h2 className="text-xl font-semibold mb-4">Upload Property Photos</h2>
                    <ImageUploader />
                  </div>
                </div>
              </TabsContent>
              <TabsContent value="enhance" className="mt-6">
                <div className="grid gap-6 md:grid-cols-2">
                  <div className="rounded-lg border p-6">
                    <h2 className="text-xl font-semibold mb-4">Enhancement Options</h2>
                    <EnhancementOptions />
                  </div>
                  <div className="rounded-lg border p-6">
                    <h2 className="text-xl font-semibold mb-4">Preview</h2>
                    <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                      <p className="text-gray-500">Select an image to preview</p>
                    </div>
                    <div className="mt-4 flex justify-end">
                      <Button className="bg-rose-500 hover:bg-rose-600">Process All Images</Button>
                    </div>
                  </div>
                </div>
              </TabsContent>
              <TabsContent value="results" className="mt-6">
                <div className="rounded-lg border p-6">
                  <h2 className="text-xl font-semibold mb-4">Processed Images</h2>
                  <ProcessedGallery />
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </main>
    </div>
  )
}

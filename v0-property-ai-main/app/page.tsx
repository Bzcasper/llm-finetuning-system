import { MainNav } from "@/components/navigation/main-nav"
import FeatureCard from "@/components/feature-card"
import BeforeAfterSlider from "@/components/before-after-slider"
import PricingSection from "@/components/pricing-section"
import { ArrowRight, Camera, ImageIcon, Layers, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import Link from "next/link"

export default function HomePage() {
  return (
    <div className="flex flex-col min-h-screen">
      <MainNav />

      <main className="flex-1">
        {/* Hero Section */}
        <section className="py-12 sm:py-16 lg:py-20 bg-gradient-to-b from-rose-50 to-white">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid gap-8 lg:grid-cols-2 lg:gap-16 items-center">
              <div className="flex flex-col justify-center space-y-6 text-center lg:text-left">
                <div className="space-y-4">
                  <h1 className="text-3xl font-bold tracking-tight sm:text-4xl md:text-5xl lg:text-6xl">
                    Transform Property Photos with AI
                  </h1>
                  <p className="text-lg text-gray-600 sm:text-xl max-w-2xl mx-auto lg:mx-0">
                    Enhance your real estate listings with professional-grade photos in seconds. No professional
                    equipment needed.
                  </p>
                </div>
                <div className="flex flex-col sm:flex-row gap-4 justify-center lg:justify-start">
                  <Button asChild size="lg" className="bg-rose-500 hover:bg-rose-600 text-base px-8 py-3">
                    <Link href="/auth/signup">
                      Get Started <ArrowRight className="ml-2 h-5 w-5" />
                    </Link>
                  </Button>
                  <Button asChild size="lg" variant="outline" className="text-base px-8 py-3">
                    <Link href="#demo">See Demo</Link>
                  </Button>
                </div>
              </div>
              <div className="flex justify-center lg:justify-end">
                <div className="w-full max-w-lg lg:max-w-none">
                  <BeforeAfterSlider
                    beforeImage="/placeholder.svg?height=600&width=800"
                    afterImage="/placeholder.svg?height=600&width=800"
                    className="rounded-lg shadow-xl w-full"
                  />
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="py-12 sm:py-16 lg:py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center space-y-4 mb-12 lg:mb-16">
              <h2 className="text-3xl font-bold tracking-tight sm:text-4xl md:text-5xl">
                Powerful AI Enhancement Features
              </h2>
              <p className="text-lg text-gray-600 sm:text-xl max-w-3xl mx-auto">
                Our AI-powered platform transforms ordinary property photos into stunning marketing assets.
              </p>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8 max-w-6xl mx-auto">
              <FeatureCard
                icon={<Sparkles className="h-10 w-10 text-rose-500" />}
                title="HDR Enhancement"
                description="Automatically fix poor lighting conditions and enhance details in shadows and highlights."
              />
              <FeatureCard
                icon={<ImageIcon className="h-10 w-10 text-rose-500" />}
                title="Perspective Correction"
                description="Fix distorted room angles and create more appealing spatial representations."
              />
              <FeatureCard
                icon={<Layers className="h-10 w-10 text-rose-500" />}
                title="Virtual Staging"
                description="Add customizable furniture sets to empty rooms to help buyers visualize the space."
              />
              <FeatureCard
                icon={<Camera className="h-10 w-10 text-rose-500" />}
                title="Sky Replacement"
                description="Replace dull skies with beautiful blue skies or dramatic sunsets for exterior shots."
              />
              <FeatureCard
                icon={<ImageIcon className="h-10 w-10 text-rose-500" />}
                title="Object Removal"
                description="Remove unwanted objects like cars, trash bins, or power lines from exterior photos."
              />
              <FeatureCard
                icon={<Sparkles className="h-10 w-10 text-rose-500" />}
                title="Batch Processing"
                description="Process multiple images at once with consistent enhancement profiles."
              />
            </div>
          </div>
        </section>

        {/* Demo Section */}
        <section id="demo" className="py-12 sm:py-16 lg:py-20 bg-gray-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center space-y-4 mb-12 lg:mb-16">
              <h2 className="text-3xl font-bold tracking-tight sm:text-4xl md:text-5xl">See the Difference</h2>
              <p className="text-lg text-gray-600 sm:text-xl max-w-3xl mx-auto">
                Drag the slider to compare before and after transformations.
              </p>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-12 max-w-6xl mx-auto">
              <div className="space-y-4">
                <BeforeAfterSlider
                  beforeImage="/placeholder.svg?height=600&width=800"
                  afterImage="/placeholder.svg?height=600&width=800"
                  className="rounded-lg shadow-xl w-full"
                />
                <p className="text-center text-sm text-gray-500">Interior - HDR Enhancement & Virtual Staging</p>
              </div>
              <div className="space-y-4">
                <BeforeAfterSlider
                  beforeImage="/placeholder.svg?height=600&width=800"
                  afterImage="/placeholder.svg?height=600&width=800"
                  className="rounded-lg shadow-xl w-full"
                />
                <p className="text-center text-sm text-gray-500">Exterior - Sky Replacement & Lawn Enhancement</p>
              </div>
            </div>
          </div>
        </section>

        {/* Pricing Section */}
        <section id="pricing" className="py-12 sm:py-16 lg:py-20">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <PricingSection />
          </div>
        </section>
      </main>

      <footer className="border-t bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 py-6 sm:py-8">
            <p className="text-center sm:text-left text-sm text-gray-500">© 2025 PropertyGlow. All rights reserved.</p>
            <div className="flex items-center gap-6">
              <Link href="/terms" className="text-sm text-gray-500 hover:text-gray-900 transition-colors">
                Terms
              </Link>
              <Link href="/privacy" className="text-sm text-gray-500 hover:text-gray-900 transition-colors">
                Privacy
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}

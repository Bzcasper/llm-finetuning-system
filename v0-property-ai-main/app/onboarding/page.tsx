"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { useSession } from "next-auth/react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Card, CardContent } from "@/components/ui/card"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Quote, Loader2, AlertCircle } from "lucide-react"
import { Alert, AlertDescription } from "@/components/ui/alert"

interface OnboardingData {
  fullName: string
  companyName: string
  userType: string
  monthlyVolume: string
  discoverySource: string
}

export default function OnboardingPage() {
  const router = useRouter()
  const { data: session, status } = useSession()
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [formData, setFormData] = useState<OnboardingData>({
    fullName: session?.user?.name || "",
    companyName: "",
    userType: "",
    monthlyVolume: "",
    discoverySource: "",
  })

  // Redirect if not authenticated
  if (status === "loading") {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 animate-spin" />
      </div>
    )
  }

  if (status === "unauthenticated") {
    router.push("/auth/signin")
    return null
  }

  const handleInputChange = (field: keyof OnboardingData, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }))
    setError("") // Clear error when user starts typing
  }

  const handleSubmit = async () => {
    setIsLoading(true)
    setError("")

    try {
      console.log("Submitting onboarding data:", formData)

      const response = await fetch("/api/onboarding", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      })

      const result = await response.json()
      console.log("API Response:", result)

      if (!response.ok) {
        throw new Error(result.error || "Failed to save onboarding data")
      }

      // Success - navigate to dashboard
      console.log("Onboarding completed successfully, navigating to dashboard")
      router.push("/dashboard")
    } catch (error) {
      console.error("Error saving onboarding data:", error)
      setError(error instanceof Error ? error.message : "An unexpected error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  const isFormValid = Object.values(formData).every((value) => value.trim() !== "")

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-8 items-center">
            {/* Left Side - Form */}
            <div className="space-y-8">
              <div className="text-center lg:text-left">
                <h1 className="text-3xl lg:text-4xl font-bold text-[#0A2342] mb-4">
                  Let's Optimize PropertyGlow For You
                </h1>
                <p className="text-lg text-[#2E5077] leading-relaxed">
                  Welcome {session?.user?.name}! It would be great to hear more about you so we can optimize
                  PropertyGlow for your specific needs.
                </p>
              </div>

              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              <div className="space-y-6">
                <div className="space-y-2">
                  <Label htmlFor="fullName" className="text-[#0A2342] font-medium">
                    What is your full name? *
                  </Label>
                  <Input
                    id="fullName"
                    placeholder="Enter your full name"
                    value={formData.fullName}
                    onChange={(e) => handleInputChange("fullName", e.target.value)}
                    className="h-12 border-2 border-slate-200 focus:border-[#F5A623] transition-colors"
                    disabled={isLoading}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="companyName" className="text-[#0A2342] font-medium">
                    What is the name of your company? *
                  </Label>
                  <Input
                    id="companyName"
                    placeholder="Enter your company name"
                    value={formData.companyName}
                    onChange={(e) => handleInputChange("companyName", e.target.value)}
                    className="h-12 border-2 border-slate-200 focus:border-[#F5A623] transition-colors"
                    disabled={isLoading}
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-[#0A2342] font-medium">How will you use PropertyGlow? *</Label>
                  <Select onValueChange={(value) => handleInputChange("userType", value)} disabled={isLoading}>
                    <SelectTrigger className="h-12 border-2 border-slate-200 focus:border-[#F5A623]">
                      <SelectValue placeholder="Select your role" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="real-estate-agent">As a real estate agent</SelectItem>
                      <SelectItem value="photographer">As a real estate photographer</SelectItem>
                      <SelectItem value="broker">As a broker/agency owner</SelectItem>
                      <SelectItem value="property-manager">As a property manager</SelectItem>
                      <SelectItem value="developer">As a property developer</SelectItem>
                      <SelectItem value="other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-[#0A2342] font-medium">
                    How many images do you enhance monthly on average? *
                  </Label>
                  <Select onValueChange={(value) => handleInputChange("monthlyVolume", value)} disabled={isLoading}>
                    <SelectTrigger className="h-12 border-2 border-slate-200 focus:border-[#F5A623]">
                      <SelectValue placeholder="Select monthly volume" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1-10">1-10 images</SelectItem>
                      <SelectItem value="11-50">11-50 images</SelectItem>
                      <SelectItem value="51-100">51-100 images</SelectItem>
                      <SelectItem value="101-250">101-250 images</SelectItem>
                      <SelectItem value="251-500">251-500 images</SelectItem>
                      <SelectItem value="500+">500+ images</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label className="text-[#0A2342] font-medium">How did you find out about PropertyGlow? *</Label>
                  <Select onValueChange={(value) => handleInputChange("discoverySource", value)} disabled={isLoading}>
                    <SelectTrigger className="h-12 border-2 border-slate-200 focus:border-[#F5A623]">
                      <SelectValue placeholder="Select discovery source" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="search">Search engine (Google, Bing)</SelectItem>
                      <SelectItem value="social-media">Social media</SelectItem>
                      <SelectItem value="referral">Referral from colleague</SelectItem>
                      <SelectItem value="advertisement">Online advertisement</SelectItem>
                      <SelectItem value="real-estate-forum">Real estate forum/community</SelectItem>
                      <SelectItem value="other">Other</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button
                  onClick={handleSubmit}
                  disabled={!isFormValid || isLoading}
                  className="w-full h-12 bg-[#0A2342] hover:bg-[#2E5077] text-white font-semibold text-lg transition-colors disabled:opacity-50"
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Saving...
                    </>
                  ) : (
                    "Continue to Dashboard"
                  )}
                </Button>

                {/* Debug Info */}
                <div className="text-xs text-gray-500 space-y-1">
                  <p>Form Valid: {isFormValid ? "Yes" : "No"}</p>
                  <p>User: {session?.user?.email}</p>
                  <p>Loading: {isLoading ? "Yes" : "No"}</p>
                </div>
              </div>
            </div>

            {/* Right Side - Testimonial Image */}
            <div className="relative">
              <Card className="overflow-hidden border-0 shadow-2xl">
                <CardContent className="p-0 relative">
                  <div className="aspect-[4/3] bg-gradient-to-br from-slate-100 to-slate-200 relative overflow-hidden">
                    {/* Background Image */}
                    <img
                      src="/placeholder.svg?height=600&width=800"
                      alt="Modern property interior"
                      className="w-full h-full object-cover"
                    />

                    {/* Overlay Gradient */}
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />

                    {/* Testimonial Overlay */}
                    <div className="absolute bottom-0 left-0 right-0 p-6 text-white">
                      <div className="flex items-start space-x-4">
                        <Avatar className="w-16 h-16 border-4 border-white/20">
                          <AvatarImage src="/placeholder-user.jpg" alt="Sarah Mitchell" />
                          <AvatarFallback className="bg-[#F5A623] text-white font-bold">SM</AvatarFallback>
                        </Avatar>
                        <div className="flex-1">
                          <p className="text-sm font-medium text-white/90 mb-1">
                            Sarah Mitchell - Mitchell Real Estate Group
                          </p>
                          <div className="relative">
                            <Quote className="absolute -top-2 -left-1 w-6 h-6 text-[#F5A623] opacity-60" />
                            <blockquote className="text-lg font-medium leading-relaxed pl-6">
                              PropertyGlow has been far more beneficial than we expected... I hope to be using this
                              service for the foreseeable future!
                            </blockquote>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Floating Stats */}
              <div className="absolute -top-4 -right-4 bg-white rounded-xl shadow-lg p-4 border border-slate-200">
                <div className="text-center">
                  <div className="text-2xl font-bold text-[#0A2342]">2.5M+</div>
                  <div className="text-sm text-[#2E5077]">Images Enhanced</div>
                </div>
              </div>

              <div className="absolute -bottom-4 -left-4 bg-white rounded-xl shadow-lg p-4 border border-slate-200">
                <div className="text-center">
                  <div className="text-2xl font-bold text-[#F5A623]">98%</div>
                  <div className="text-sm text-[#2E5077]">Satisfaction Rate</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

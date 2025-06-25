import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Check } from "lucide-react"

const plans = [
  {
    name: "Solo Agent",
    price: "$49",
    description: "Perfect for individual real estate agents",
    features: [
      "5 property enhancements per month",
      "Basic HDR enhancement",
      "Sky replacement",
      "Email support",
      "Standard processing speed",
    ],
  },
  {
    name: "Team",
    price: "$149",
    description: "Ideal for small real estate teams",
    features: [
      "25 property enhancements per month",
      "All enhancement features",
      "Virtual staging",
      "Priority support",
      "Batch processing",
      "Team collaboration tools",
    ],
    popular: true,
  },
  {
    name: "Brokerage",
    price: "$499",
    description: "For large brokerages and agencies",
    features: [
      "100 property enhancements per month",
      "All premium features",
      "Custom branding",
      "Dedicated account manager",
      "API access",
      "Advanced analytics",
      "White-label solution",
    ],
  },
]

export default function PricingSection() {
  return (
    <div className="space-y-8 sm:space-y-12">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold tracking-tight sm:text-4xl md:text-5xl">Simple, Transparent Pricing</h2>
        <p className="text-lg text-gray-600 sm:text-xl max-w-3xl mx-auto">
          Choose the perfect plan for your real estate business. All plans include our core AI enhancement features.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8 max-w-6xl mx-auto">
        {plans.map((plan, index) => (
          <Card
            key={index}
            className={`relative h-full ${plan.popular ? "border-rose-500 shadow-lg scale-105" : "border-gray-200"}`}
          >
            {plan.popular && (
              <div className="absolute -top-3 left-1/2 transform -translate-x-1/2">
                <span className="bg-rose-500 text-white px-4 py-1 rounded-full text-sm font-medium">Most Popular</span>
              </div>
            )}

            <CardHeader className="text-center pb-4">
              <CardTitle className="text-xl sm:text-2xl">{plan.name}</CardTitle>
              <CardDescription className="text-sm sm:text-base">{plan.description}</CardDescription>
              <div className="pt-4">
                <span className="text-3xl sm:text-4xl font-bold">{plan.price}</span>
                <span className="text-gray-600 ml-1">/month</span>
              </div>
            </CardHeader>

            <CardContent className="space-y-3">
              {plan.features.map((feature, featureIndex) => (
                <div key={featureIndex} className="flex items-start gap-3">
                  <Check className="h-5 w-5 text-green-500 mt-0.5 flex-shrink-0" />
                  <span className="text-sm sm:text-base text-gray-700">{feature}</span>
                </div>
              ))}
            </CardContent>

            <CardFooter className="pt-6">
              <Button
                className={`w-full ${plan.popular ? "bg-rose-500 hover:bg-rose-600" : "bg-gray-900 hover:bg-gray-800"}`}
                size="lg"
              >
                Get Started
              </Button>
            </CardFooter>
          </Card>
        ))}
      </div>

      <div className="text-center">
        <p className="text-sm text-gray-600">All plans include a 14-day free trial. No credit card required.</p>
      </div>
    </div>
  )
}

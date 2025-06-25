"use client"

import { useState } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
import { Home, Menu } from "lucide-react"
import { cn } from "@/lib/utils"

const navigation = [
  { name: "Features", href: "#features", type: "scroll" },
  { name: "Demo", href: "#demo", type: "scroll" },
  { name: "Pricing", href: "#pricing", type: "scroll" },
  { name: "Dashboard", href: "/dashboard", type: "link" },
]

export function MainNav() {
  const [isOpen, setIsOpen] = useState(false)
  const pathname = usePathname()

  const handleScrollTo = (elementId: string) => {
    const element = document.getElementById(elementId.replace("#", ""))
    if (element) {
      element.scrollIntoView({ behavior: "smooth" })
      setIsOpen(false)
    }
  }

  const handleNavClick = (item: (typeof navigation)[0]) => {
    if (item.type === "scroll" && pathname === "/") {
      handleScrollTo(item.href)
    }
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 hover:opacity-80 transition-opacity">
            <Home className="h-6 w-6 text-rose-500" />
            <span className="text-xl font-bold text-gray-900">PropertyGlow</span>
          </Link>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center gap-8">
            {navigation.map((item) => (
              <button
                key={item.name}
                onClick={() => handleNavClick(item)}
                className={cn(
                  "text-sm font-medium transition-colors hover:text-rose-500",
                  pathname === item.href ? "text-rose-500" : "text-gray-700",
                )}
              >
                {item.name}
              </button>
            ))}
          </nav>

          {/* Desktop Auth Buttons */}
          <div className="hidden md:flex items-center gap-3">
            <Button variant="outline" size="sm" asChild>
              <Link href="/auth/signin">Sign In</Link>
            </Button>
            <Button size="sm" className="bg-rose-500 hover:bg-rose-600" asChild>
              <Link href="/auth/signup">Get Started</Link>
            </Button>
          </div>

          {/* Mobile Menu */}
          <Sheet open={isOpen} onOpenChange={setIsOpen}>
            <SheetTrigger asChild className="md:hidden">
              <Button variant="ghost" size="sm">
                <Menu className="h-5 w-5" />
                <span className="sr-only">Toggle menu</span>
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-80">
              <div className="flex flex-col h-full">
                <div className="flex items-center justify-between pb-4 border-b">
                  <Link href="/" className="flex items-center gap-2" onClick={() => setIsOpen(false)}>
                    <Home className="h-6 w-6 text-rose-500" />
                    <span className="text-xl font-bold">PropertyGlow</span>
                  </Link>
                </div>

                <nav className="flex flex-col gap-4 py-6">
                  {navigation.map((item) => (
                    <button
                      key={item.name}
                      onClick={() => handleNavClick(item)}
                      className="text-left text-lg font-medium text-gray-700 hover:text-rose-500 transition-colors"
                    >
                      {item.name}
                    </button>
                  ))}
                </nav>

                <div className="mt-auto space-y-3">
                  <Button variant="outline" className="w-full" asChild>
                    <Link href="/auth/signin" onClick={() => setIsOpen(false)}>
                      Sign In
                    </Link>
                  </Button>
                  <Button className="w-full bg-rose-500 hover:bg-rose-600" asChild>
                    <Link href="/auth/signup" onClick={() => setIsOpen(false)}>
                      Get Started
                    </Link>
                  </Button>
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  )
}

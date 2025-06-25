import { NextResponse } from "next/server"
import { checkEnvironmentHealth, validateEnvGroup } from "@/lib/env-validation"
import { isEnvironmentReady, getValidationError } from "@/lib/env-loader"
import { getConfigurationStatus } from "@/lib/config"

export async function GET() {
  try {
    // Get comprehensive environment health check
    const health = checkEnvironmentHealth()
    const configStatus = getConfigurationStatus()
    const validationError = getValidationError()
    
    // Test individual service configurations
    const serviceTests = {
      database: testService(() => validateEnvGroup.database(), "Database connection"),
      auth: testService(() => validateEnvGroup.auth(), "Authentication"),
      ai: testService(() => validateEnvGroup.ai(), "AI services"),
      storage: testService(() => validateEnvGroup.storage(), "File storage"),
      payments: testService(() => validateEnvGroup.payments(), "Payment processing")
    }
    
    // Determine overall status
    const isReady = health.isValid && isEnvironmentReady()
    const criticalErrors = health.errors.filter(error => 
      !error.includes("SENTRY_DSN") && 
      !error.includes("ANALYTICS_ID") &&
      !error.includes("MAPBOX_API_KEY")
    )
    
    const response = {
      ready: isReady,
      environment: health.environment,
      status: isReady ? "healthy" : "unhealthy",
      timestamp: new Date().toISOString(),
      
      // Overall validation results
      validation: {
        isValid: health.isValid,
        hasErrors: health.errors.length > 0,
        hasCriticalErrors: criticalErrors.length > 0,
        errorCount: health.errors.length,
        criticalErrorCount: criticalErrors.length
      },
      
      // Service-specific status
      services: {
        database: {
          configured: configStatus.database,
          status: serviceTests.database.status,
          message: serviceTests.database.message
        },
        authentication: {
          configured: configStatus.auth,
          status: serviceTests.auth.status,
          message: serviceTests.auth.message
        },
        ai: {
          configured: configStatus.ai,
          status: serviceTests.ai.status,
          message: serviceTests.ai.message
        },
        storage: {
          configured: configStatus.storage,
          status: serviceTests.storage.status,
          message: serviceTests.storage.message
        },
        payments: {
          configured: configStatus.payments,
          status: serviceTests.payments.status,
          message: serviceTests.payments.message
        },
        monitoring: {
          configured: configStatus.monitoring,
          status: configStatus.monitoring ? "configured" : "optional",
          message: configStatus.monitoring ? "Sentry monitoring enabled" : "Optional - recommended for production"
        },
        analytics: {
          configured: configStatus.analytics,
          status: configStatus.analytics ? "configured" : "optional",
          message: configStatus.analytics ? "Analytics tracking enabled" : "Optional - for usage insights"
        }
      },
      
      // Error details (only include if there are errors)
      ...(health.errors.length > 0 && {
        errors: health.errors,
        criticalErrors: criticalErrors
      }),
      
      // Validation error details (if environment validation failed)
      ...(validationError && {
        validationError: {
          message: validationError.message,
          missingVariables: validationError.getMissingVariables(),
          errorDetails: validationError.errors.map(err => ({
            field: err.field,
            message: err.message,
            code: err.code
          }))
        }
      }),
      
      // Helpful information for debugging
      debugging: {
        nodeEnv: process.env.NODE_ENV || "development",
        platform: process.platform,
        nodeVersion: process.version,
        hasEnvFile: process.env.NODE_ENV === "development" ? "Unknown (check .env file)" : "N/A",
        recommendedActions: getRecommendedActions(health.errors, criticalErrors)
      }
    }
    
    // Set appropriate HTTP status code
    const statusCode = isReady ? 200 : (criticalErrors.length > 0 ? 503 : 206)
    
    return NextResponse.json(response, { status: statusCode })
    
  } catch (error) {
    // Handle unexpected errors
    console.error("Environment check failed:", error)
    
    return NextResponse.json({
      ready: false,
      status: "error",
      environment: process.env.NODE_ENV || "unknown",
      timestamp: new Date().toISOString(),
      error: {
        message: error instanceof Error ? error.message : "Unknown error during environment check",
        type: "unexpected_error"
      },
      debugging: {
        nodeEnv: process.env.NODE_ENV || "development",
        recommendation: "Check server logs for detailed error information"
      }
    }, { status: 500 })
  }
}

/**
 * Test a specific service configuration
 */
function testService(testFn: () => any, serviceName: string) {
  try {
    testFn()
    return {
      status: "healthy" as const,
      message: `${serviceName} is properly configured`
    }
  } catch (error) {
    return {
      status: "error" as const,
      message: error instanceof Error ? error.message : `${serviceName} configuration error`
    }
  }
}

/**
 * Generate recommended actions based on errors
 */
function getRecommendedActions(errors: string[], criticalErrors: string[]): string[] {
  const actions: string[] = []
  
  if (criticalErrors.length > 0) {
    actions.push("🚨 Fix critical environment variables to enable core functionality")
  }
  
  if (errors.some(err => err.includes("DATABASE_URL"))) {
    actions.push("📊 Configure DATABASE_URL for data persistence")
  }
  
  if (errors.some(err => err.includes("NEXTAUTH"))) {
    actions.push("🔐 Set up NextAuth configuration for user authentication")
  }
  
  if (errors.some(err => err.includes("GOOGLE_CLIENT"))) {
    actions.push("🔑 Configure Google OAuth credentials")
  }
  
  if (errors.some(err => err.includes("FAL_KEY"))) {
    actions.push("🤖 Add FAL_KEY for AI image processing capabilities")
  }
  
  if (errors.some(err => err.includes("STRIPE"))) {
    actions.push("💳 Configure Stripe keys for payment processing")
  }
  
  if (errors.some(err => err.includes("BLOB_READ_WRITE_TOKEN"))) {
    actions.push("📁 Set up Vercel Blob storage for file handling")
  }
  
  if (process.env.NODE_ENV === "production" && errors.some(err => err.includes("SENTRY_DSN"))) {
    actions.push("📊 Consider adding Sentry for production error monitoring")
  }
  
  if (actions.length === 0 && errors.length > 0) {
    actions.push("✅ Review optional configurations to enable additional features")
  }
  
  if (actions.length === 0) {
    actions.push("🎉 All configurations look good!")
  }
  
  return actions
}

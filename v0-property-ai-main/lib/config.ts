import { getEnvironment, getEnvVar, isEnvironmentReady } from "./env-loader"
import { checkEnvironmentHealth } from "./env-validation"

/**
 * Type-safe environment configuration for PropertyGlow
 * Uses the validated environment variables from env-loader
 */
export const config = {
  // Application environment
  get nodeEnv() {
    return getEnvVar("NODE_ENV", "development")
  },
  
  // Database configuration
  database: {
    get url() {
      return getEnvVar("DATABASE_URL")
    }
  },
  
  // Authentication configuration
  auth: {
    get secret() {
      return getEnvVar("NEXTAUTH_SECRET")
    },
    get url() {
      return getEnvVar("NEXTAUTH_URL")
    },
    google: {
      get clientId() {
        return getEnvVar("GOOGLE_CLIENT_ID")
      },
      get clientSecret() {
        return getEnvVar("GOOGLE_CLIENT_SECRET")
      }
    }
  },
  
  // AI/ML services configuration
  ai: {
    fal: {
      get key() {
        return getEnvVar("FAL_KEY")
      }
    }
  },
  
  // File storage configuration
  storage: {
    blob: {
      get token() {
        return getEnvVar("BLOB_READ_WRITE_TOKEN")
      }
    }
  },
  
  // Payment processing configuration
  payments: {
    stripe: {
      get secretKey() {
        return getEnvVar("STRIPE_SECRET_KEY")
      },
      get publishableKey() {
        return getEnvVar("STRIPE_PUBLISHABLE_KEY")
      },
      get webhookSecret() {
        return getEnvVar("STRIPE_WEBHOOK_SECRET")
      }
    }
  },
  
  // Optional services configuration
  monitoring: {
    get sentryDsn() {
      return getEnvVar("SENTRY_DSN")
    }
  },
  
  analytics: {
    get id() {
      return getEnvVar("ANALYTICS_ID")
    }
  },
  
  // Third-party APIs (optional)
  mapbox: {
    get apiKey() {
      return getEnvVar("MAPBOX_API_KEY")
    }
  },
  
  // Deployment configuration
  deployment: {
    get vercelUrl() {
      return getEnvVar("VERCEL_URL")
    }
  }
}

/**
 * Legacy helper function for backward compatibility
 * @deprecated Use the new validation system from env-validation.ts
 */
export function validateEnvVars(): boolean {
  try {
    if (!isEnvironmentReady()) {
      return false
    }
    
    const health = checkEnvironmentHealth()
    return health.isValid
  } catch {
    return false
  }
}

/**
 * Get validated environment configuration
 * @returns Validated environment object
 */
export function getValidatedConfig() {
  return getEnvironment()
}

/**
 * Check if a specific service is configured
 */
export const isServiceConfigured = {
  database: () => !!config.database.url,
  auth: () => !!(config.auth.secret && config.auth.google.clientId && config.auth.google.clientSecret),
  ai: () => !!config.ai.fal.key,
  storage: () => !!config.storage.blob.token,
  payments: () => !!(config.payments.stripe.secretKey && config.payments.stripe.publishableKey && config.payments.stripe.webhookSecret),
  monitoring: () => !!config.monitoring.sentryDsn,
  analytics: () => !!config.analytics.id,
  mapbox: () => !!config.mapbox.apiKey
}

/**
 * Get configuration status for all services
 */
export function getConfigurationStatus() {
  return {
    database: isServiceConfigured.database(),
    auth: isServiceConfigured.auth(),
    ai: isServiceConfigured.ai(),
    storage: isServiceConfigured.storage(),
    payments: isServiceConfigured.payments(),
    monitoring: isServiceConfigured.monitoring(),
    analytics: isServiceConfigured.analytics(),
    mapbox: isServiceConfigured.mapbox(),
    overall: isEnvironmentReady()
  }
}

// Server-side environment status logging
if (typeof window === "undefined") {
  // Use a timeout to ensure env-loader has had a chance to initialize
  setTimeout(() => {
    try {
      const status = getConfigurationStatus()
      console.log("🔧 PropertyGlow Configuration Status:")
      console.log(`   Database: ${status.database ? "✅ Configured" : "❌ Missing"}`)
      console.log(`   Authentication: ${status.auth ? "✅ Configured" : "❌ Missing"}`)
      console.log(`   AI Services: ${status.ai ? "✅ Configured" : "❌ Missing"}`)
      console.log(`   Storage: ${status.storage ? "✅ Configured" : "❌ Missing"}`)
      console.log(`   Payments: ${status.payments ? "✅ Configured" : "❌ Missing"}`)
      console.log(`   Monitoring: ${status.monitoring ? "✅ Configured" : "⚠️  Optional"}`)
      console.log(`   Analytics: ${status.analytics ? "✅ Configured" : "⚠️  Optional"}`)
      console.log(`   Mapbox: ${status.mapbox ? "✅ Configured" : "⚠️  Optional"}`)
      console.log(`   Overall Status: ${status.overall ? "✅ Ready" : "❌ Issues Found"}`)
    } catch (error) {
      console.log("⚠️  Configuration status check skipped - environment validation in progress")
    }
  }, 100)
}

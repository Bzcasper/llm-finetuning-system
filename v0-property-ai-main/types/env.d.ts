/**
 * TypeScript environment variable declarations
 * Provides type safety for all environment variables used in PropertyGlow
 */

declare global {
  namespace NodeJS {
    interface ProcessEnv {
      // Node.js environment
      readonly NODE_ENV: "development" | "staging" | "production"
      
      // Database configuration
      readonly DATABASE_URL: string
      
      // NextAuth.js configuration
      readonly NEXTAUTH_SECRET: string
      readonly NEXTAUTH_URL: string
      
      // Google OAuth configuration
      readonly GOOGLE_CLIENT_ID: string
      readonly GOOGLE_CLIENT_SECRET: string
      
      // AI/ML services
      readonly FAL_KEY: string
      
      // File storage
      readonly BLOB_READ_WRITE_TOKEN: string
      
      // Payment processing (Stripe)
      readonly STRIPE_SECRET_KEY: string
      readonly STRIPE_PUBLISHABLE_KEY: string
      readonly STRIPE_WEBHOOK_SECRET: string
      
      // Optional: Monitoring and analytics
      readonly SENTRY_DSN?: string
      readonly ANALYTICS_ID?: string
      
      // Optional: Third-party APIs
      readonly MAPBOX_API_KEY?: string
      
      // Optional: Deployment configuration
      readonly VERCEL_URL?: string
      
      // Optional: Email configuration
      readonly SMTP_HOST?: string
      readonly SMTP_PORT?: string
      readonly SMTP_USER?: string
      readonly SMTP_PASSWORD?: string
    }
  }
}

/**
 * Environment variable configuration types
 * Used for type-safe access to validated environment variables
 */
export interface EnvironmentConfig {
  // Application environment
  nodeEnv: "development" | "staging" | "production"
  
  // Database configuration
  database: {
    url: string
  }
  
  // Authentication configuration
  auth: {
    secret: string
    url: string
    google: {
      clientId: string
      clientSecret: string
    }
  }
  
  // AI/ML services configuration
  ai: {
    fal: {
      key: string
    }
  }
  
  // File storage configuration
  storage: {
    blob: {
      token: string
    }
  }
  
  // Payment processing configuration
  payments: {
    stripe: {
      secretKey: string
      publishableKey: string
      webhookSecret: string
    }
  }
  
  // Optional monitoring configuration
  monitoring?: {
    sentryDsn?: string
  }
  
  // Optional analytics configuration
  analytics?: {
    id?: string
  }
  
  // Optional third-party APIs
  mapbox?: {
    apiKey?: string
  }
  
  // Optional deployment configuration
  deployment?: {
    vercelUrl?: string
  }
  
  // Optional email configuration
  email?: {
    smtp?: {
      host?: string
      port?: string
      user?: string
      password?: string
    }
  }
}

/**
 * Environment validation result types
 */
export interface EnvironmentValidationResult {
  isValid: boolean
  errors: string[]
  warnings: string[]
  environment: string
}

/**
 * Service configuration status types
 */
export interface ServiceConfigurationStatus {
  database: boolean
  auth: boolean
  ai: boolean
  storage: boolean
  payments: boolean
  monitoring: boolean
  analytics: boolean
  mapbox: boolean
  overall: boolean
}

/**
 * Environment health check types
 */
export interface EnvironmentHealthCheck {
  ready: boolean
  environment: string
  status: "healthy" | "unhealthy" | "error"
  timestamp: string
  validation: {
    isValid: boolean
    hasErrors: boolean
    hasCriticalErrors: boolean
    errorCount: number
    criticalErrorCount: number
  }
  services: {
    database: ServiceStatus
    authentication: ServiceStatus
    ai: ServiceStatus
    storage: ServiceStatus
    payments: ServiceStatus
    monitoring: ServiceStatus
    analytics: ServiceStatus
  }
  errors?: string[]
  criticalErrors?: string[]
  validationError?: {
    message: string
    missingVariables: string[]
    errorDetails: Array<{
      field: string
      message: string
      code: string
    }>
  }
  debugging: {
    nodeEnv: string
    platform: string
    nodeVersion: string
    hasEnvFile: string
    recommendedActions: string[]
  }
}

export interface ServiceStatus {
  configured: boolean
  status: "healthy" | "error" | "configured" | "optional"
  message: string
}

/**
 * Environment-specific validation schema types
 */
export type EnvironmentType = "development" | "staging" | "production"

export interface EnvironmentValidationOptions {
  failFast?: boolean
  logLevel?: "silent" | "error" | "info" | "debug"
}

/**
 * Stripe configuration types (environment-specific)
 */
export interface StripeConfig {
  secretKey: string
  publishableKey: string
  webhookSecret: string
  isLiveMode: boolean
}

/**
 * Re-export the main environment schema type from validation
 */
export type { EnvSchema } from "@/lib/env-validation"

// This export is required for the global type declaration to work
export {}
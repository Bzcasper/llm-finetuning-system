import { z } from "zod"

/**
 * Environment validation schema using Zod
 * Provides comprehensive validation for all required environment variables
 */

// Base environment validation schema
const baseEnvSchema = z.object({
  // Node environment
  NODE_ENV: z.enum(["development", "staging", "production"]).default("development"),
  
  // Database
  DATABASE_URL: z.string().url("DATABASE_URL must be a valid PostgreSQL connection string"),
  
  // NextAuth configuration
  NEXTAUTH_SECRET: z.string().min(32, "NEXTAUTH_SECRET must be at least 32 characters long for security"),
  NEXTAUTH_URL: z.string().url("NEXTAUTH_URL must be a valid URL"),
  
  // Google OAuth
  GOOGLE_CLIENT_ID: z.string().min(1, "GOOGLE_CLIENT_ID is required for OAuth authentication"),
  GOOGLE_CLIENT_SECRET: z.string().min(1, "GOOGLE_CLIENT_SECRET is required for OAuth authentication"),
  
  // FAL AI API
  FAL_KEY: z.string().min(1, "FAL_KEY is required for AI image processing"),
  
  // Vercel Blob Storage
  BLOB_READ_WRITE_TOKEN: z.string().min(1, "BLOB_READ_WRITE_TOKEN is required for file storage"),
  
  // Stripe Payment Processing
  STRIPE_SECRET_KEY: z.string().startsWith("sk_", "STRIPE_SECRET_KEY must start with 'sk_'"),
  STRIPE_PUBLISHABLE_KEY: z.string().startsWith("pk_", "STRIPE_PUBLISHABLE_KEY must start with 'pk_'"),
  STRIPE_WEBHOOK_SECRET: z.string().startsWith("whsec_", "STRIPE_WEBHOOK_SECRET must start with 'whsec_'"),
})

// Optional environment variables (for enhanced features)
const optionalEnvSchema = z.object({
  // Optional: Monitoring and Analytics
  SENTRY_DSN: z.string().url().optional(),
  ANALYTICS_ID: z.string().optional(),
  
  // Optional: Additional API keys
  MAPBOX_API_KEY: z.string().optional(),
  
  // Optional: Vercel deployment URL
  VERCEL_URL: z.string().url().optional(),
  
  // Optional: Email configuration
  SMTP_HOST: z.string().optional(),
  SMTP_PORT: z.string().optional(),
  SMTP_USER: z.string().optional(),
  SMTP_PASSWORD: z.string().optional(),
})

// Environment-specific validation schemas
const developmentEnvSchema = baseEnvSchema.extend({
  NODE_ENV: z.literal("development"),
  // In development, some keys can be optional or have different validation
  STRIPE_SECRET_KEY: z.string().startsWith("sk_test_", "Development STRIPE_SECRET_KEY must start with 'sk_test_'"),
  STRIPE_PUBLISHABLE_KEY: z.string().startsWith("pk_test_", "Development STRIPE_PUBLISHABLE_KEY must start with 'pk_test_'"),
}).merge(optionalEnvSchema)

const stagingEnvSchema = baseEnvSchema.extend({
  NODE_ENV: z.literal("staging"),
  // Staging should use test keys but validate more strictly
  STRIPE_SECRET_KEY: z.string().startsWith("sk_test_", "Staging STRIPE_SECRET_KEY must start with 'sk_test_'"),
  STRIPE_PUBLISHABLE_KEY: z.string().startsWith("pk_test_", "Staging STRIPE_PUBLISHABLE_KEY must start with 'pk_test_'"),
}).merge(optionalEnvSchema)

const productionEnvSchema = baseEnvSchema.extend({
  NODE_ENV: z.literal("production"),
  // Production must use live keys and have stricter validation
  STRIPE_SECRET_KEY: z.string().startsWith("sk_live_", "Production STRIPE_SECRET_KEY must start with 'sk_live_'"),
  STRIPE_PUBLISHABLE_KEY: z.string().startsWith("pk_live_", "Production STRIPE_PUBLISHABLE_KEY must start with 'pk_live_'"),
  SENTRY_DSN: z.string().url("SENTRY_DSN is required in production for error monitoring"),
}).merge(optionalEnvSchema)

// Combined schema type
export type EnvSchema = z.infer<typeof baseEnvSchema> & z.infer<typeof optionalEnvSchema>

/**
 * Validates environment variables based on the current NODE_ENV
 * @param env - Environment variables object (defaults to process.env)
 * @returns Parsed and validated environment variables
 */
export function validateEnvironment(env: Record<string, string | undefined> = process.env): EnvSchema {
  const nodeEnv = env.NODE_ENV || "development"
  
  let schema: z.ZodType<any>
  
  switch (nodeEnv) {
    case "production":
      schema = productionEnvSchema
      break
    case "staging":
      schema = stagingEnvSchema
      break
    case "development":
    default:
      schema = developmentEnvSchema
      break
  }
  
  try {
    return schema.parse(env)
  } catch (error) {
    if (error instanceof z.ZodError) {
      const missingVars = error.errors.map(err => ({
        field: err.path.join('.'),
        message: err.message,
        code: err.code
      }))
      
      throw new EnvironmentValidationError(
        `Environment validation failed for ${nodeEnv} environment`,
        missingVars
      )
    }
    throw error
  }
}

/**
 * Custom error class for environment validation failures
 */
export class EnvironmentValidationError extends Error {
  constructor(
    message: string,
    public readonly errors: Array<{
      field: string
      message: string
      code: string
    }>
  ) {
    super(message)
    this.name = "EnvironmentValidationError"
  }
  
  /**
   * Returns a formatted error message for logging
   */
  getFormattedMessage(): string {
    const errorDetails = this.errors
      .map(err => `  • ${err.field}: ${err.message}`)
      .join('\n')
    
    return `${this.message}:\n${errorDetails}`
  }
  
  /**
   * Returns environment variables that need to be set
   */
  getMissingVariables(): string[] {
    return this.errors
      .filter(err => err.code === 'invalid_type' && err.message.includes('Required'))
      .map(err => err.field)
  }
}

/**
 * Validates specific environment variable groups
 */
export const validateEnvGroup = {
  /**
   * Validates database configuration
   */
  database: (env: Record<string, string | undefined> = process.env) => {
    const schema = z.object({
      DATABASE_URL: z.string().url("DATABASE_URL must be a valid PostgreSQL connection string")
    })
    return schema.parse(env)
  },
  
  /**
   * Validates authentication configuration
   */
  auth: (env: Record<string, string | undefined> = process.env) => {
    const schema = z.object({
      NEXTAUTH_SECRET: z.string().min(32, "NEXTAUTH_SECRET must be at least 32 characters long"),
      NEXTAUTH_URL: z.string().url("NEXTAUTH_URL must be a valid URL"),
      GOOGLE_CLIENT_ID: z.string().min(1, "GOOGLE_CLIENT_ID is required"),
      GOOGLE_CLIENT_SECRET: z.string().min(1, "GOOGLE_CLIENT_SECRET is required")
    })
    return schema.parse(env)
  },
  
  /**
   * Validates payment processing configuration
   */
  payments: (env: Record<string, string | undefined> = process.env) => {
    const nodeEnv = env.NODE_ENV || "development"
    const isProduction = nodeEnv === "production"
    
    const schema = z.object({
      STRIPE_SECRET_KEY: z.string().startsWith(
        isProduction ? "sk_live_" : "sk_test_",
        `STRIPE_SECRET_KEY must start with '${isProduction ? "sk_live_" : "sk_test_"}' in ${nodeEnv}`
      ),
      STRIPE_PUBLISHABLE_KEY: z.string().startsWith(
        isProduction ? "pk_live_" : "pk_test_",
        `STRIPE_PUBLISHABLE_KEY must start with '${isProduction ? "pk_live_" : "pk_test_"}' in ${nodeEnv}`
      ),
      STRIPE_WEBHOOK_SECRET: z.string().startsWith("whsec_", "STRIPE_WEBHOOK_SECRET must start with 'whsec_'")
    })
    return schema.parse(env)
  },
  
  /**
   * Validates AI/ML service configuration
   */
  ai: (env: Record<string, string | undefined> = process.env) => {
    const schema = z.object({
      FAL_KEY: z.string().min(1, "FAL_KEY is required for AI image processing")
    })
    return schema.parse(env)
  },
  
  /**
   * Validates storage configuration
   */
  storage: (env: Record<string, string | undefined> = process.env) => {
    const schema = z.object({
      BLOB_READ_WRITE_TOKEN: z.string().min(1, "BLOB_READ_WRITE_TOKEN is required for file storage")
    })
    return schema.parse(env)
  }
}

/**
 * Utility function to check if all required environment variables are set
 * without throwing errors (useful for health checks)
 */
export function checkEnvironmentHealth(env: Record<string, string | undefined> = process.env): {
  isValid: boolean
  errors: string[]
  warnings: string[]
  environment: string
} {
  const nodeEnv = env.NODE_ENV || "development"
  const errors: string[] = []
  const warnings: string[] = []
  
  try {
    validateEnvironment(env)
    return {
      isValid: true,
      errors: [],
      warnings: [],
      environment: nodeEnv
    }
  } catch (error) {
    if (error instanceof EnvironmentValidationError) {
      errors.push(...error.errors.map(err => `${err.field}: ${err.message}`))
    } else {
      errors.push(error instanceof Error ? error.message : String(error))
    }
    
    return {
      isValid: false,
      errors,
      warnings,
      environment: nodeEnv
    }
  }
}
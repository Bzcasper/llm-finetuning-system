import { validateEnvironment, EnvironmentValidationError, type EnvSchema } from "./env-validation"

/**
 * Global environment configuration loaded and validated at startup
 */
let _env: EnvSchema | null = null
let _validationError: EnvironmentValidationError | null = null

/**
 * Loads and validates environment variables at application startup
 * This function should be called as early as possible in the application lifecycle
 * 
 * @param env - Environment variables object (defaults to process.env)
 * @param options - Configuration options
 * @returns Promise that resolves when environment is validated
 */
export async function loadEnvironment(
  env: Record<string, string | undefined> = process.env,
  options: {
    failFast?: boolean
    logLevel?: 'silent' | 'error' | 'info' | 'debug'
  } = {}
): Promise<EnvSchema> {
  const { failFast = true, logLevel = 'info' } = options
  
  // Return cached environment if already loaded successfully
  if (_env) {
    return _env
  }
  
  // If we had a previous validation error and failFast is enabled, throw it immediately
  if (_validationError && failFast) {
    throw _validationError
  }
  
  try {
    if (logLevel !== 'silent') {
      console.log(`🔧 PropertyGlow: Loading environment configuration for ${env.NODE_ENV || 'development'}...`)
    }
    
    _env = validateEnvironment(env)
    _validationError = null
    
    if (logLevel === 'info' || logLevel === 'debug') {
      console.log('✅ PropertyGlow: Environment validation successful')
      
      if (logLevel === 'debug') {
        logEnvironmentStatus(_env)
      }
    }
    
    return _env
  } catch (error) {
    if (error instanceof EnvironmentValidationError) {
      _validationError = error
      
      if (logLevel !== 'silent') {
        console.error('❌ PropertyGlow: Environment validation failed')
        console.error(error.getFormattedMessage())
        
        const missingVars = error.getMissingVariables()
        if (missingVars.length > 0) {
          console.error('\n💡 To fix this issue, set the following environment variables:')
          missingVars.forEach(varName => {
            console.error(`   export ${varName}="your-value-here"`)
          })
          console.error('\n📝 Check your .env.example file for reference values')
        }
      }
      
      if (failFast) {
        console.error('\n🚨 Application startup aborted due to invalid environment configuration')
        process.exit(1)
      }
      
      throw error
    }
    
    // Handle unexpected errors
    if (logLevel !== 'silent') {
      console.error('❌ PropertyGlow: Unexpected error during environment validation:', error)
    }
    
    if (failFast) {
      process.exit(1)
    }
    
    throw error
  }
}

/**
 * Gets the validated environment configuration
 * Throws an error if environment hasn't been loaded or validation failed
 * 
 * @returns Validated environment configuration
 */
export function getEnvironment(): EnvSchema {
  if (!_env) {
    if (_validationError) {
      throw new Error(
        'Environment validation failed. Please fix the configuration errors and restart the application.\n' +
        _validationError.getFormattedMessage()
      )
    }
    
    throw new Error(
      'Environment not loaded. Call loadEnvironment() first during application startup.'
    )
  }
  
  return _env
}

/**
 * Checks if environment is loaded and valid
 * @returns boolean indicating if environment is ready
 */
export function isEnvironmentReady(): boolean {
  return _env !== null && _validationError === null
}

/**
 * Gets the last validation error (if any)
 * @returns The validation error or null if validation was successful
 */
export function getValidationError(): EnvironmentValidationError | null {
  return _validationError
}

/**
 * Resets the environment state (useful for testing)
 * @internal
 */
export function _resetEnvironment(): void {
  _env = null
  _validationError = null
}

/**
 * Logs detailed environment status for debugging
 * @private
 */
function logEnvironmentStatus(env: EnvSchema): void {
  console.log('\n🔍 PropertyGlow: Environment Configuration Status:')
  console.log(`   Environment: ${env.NODE_ENV}`)
  console.log(`   Database: ${env.DATABASE_URL ? '✅ Connected' : '❌ Not configured'}`)
  console.log(`   Authentication: ${env.NEXTAUTH_SECRET && env.GOOGLE_CLIENT_ID ? '✅ Configured' : '❌ Not configured'}`)
  console.log(`   AI Services: ${env.FAL_KEY ? '✅ Configured' : '❌ Not configured'}`)
  console.log(`   File Storage: ${env.BLOB_READ_WRITE_TOKEN ? '✅ Configured' : '❌ Not configured'}`)
  console.log(`   Payments: ${env.STRIPE_SECRET_KEY && env.STRIPE_PUBLISHABLE_KEY ? '✅ Configured' : '❌ Not configured'}`)
  console.log(`   Monitoring: ${env.SENTRY_DSN ? '✅ Configured' : '⚠️  Optional (recommended for production)'}`)
  console.log('')
}

/**
 * Environment validation middleware for API routes
 * Ensures environment is loaded before handling requests
 */
export function withEnvironmentValidation<T extends (...args: any[]) => any>(
  handler: T
): T {
  return ((...args: any[]) => {
    if (!isEnvironmentReady()) {
      const error = getValidationError()
      if (error) {
        throw new Error(`Environment validation failed: ${error.message}`)
      }
      throw new Error('Environment not initialized')
    }
    
    return handler(...args)
  }) as T
}

/**
 * Server-side environment loader that runs during application startup
 * This is automatically executed when this module is imported on the server side
 */
if (typeof window === "undefined") {
  // Only run on server-side
  const shouldFailFast = process.env.NODE_ENV === "production"
  const logLevel = process.env.NODE_ENV === "production" ? "error" : "info"
  
  // Load environment asynchronously but don't block module loading
  loadEnvironment(process.env, { 
    failFast: shouldFailFast, 
    logLevel: logLevel as any 
  }).catch(error => {
    // In development, log the error but don't crash immediately
    // This allows the developer to see the validation errors in the UI
    if (process.env.NODE_ENV !== "production") {
      console.error("⚠️  Environment validation failed - some features may not work correctly")
    }
  })
}

/**
 * Utility function to safely access environment variables with fallbacks
 */
export function getEnvVar(key: keyof EnvSchema, fallback?: string): string | undefined {
  try {
    const env = getEnvironment()
    return env[key] as string || fallback
  } catch {
    return process.env[key] || fallback
  }
}

/**
 * Type-safe environment variable accessor
 * Provides compile-time type checking for environment variable access
 */
export const env = new Proxy({} as EnvSchema, {
  get(_, prop: string | symbol) {
    if (typeof prop === 'string') {
      return getEnvVar(prop as keyof EnvSchema)
    }
    return undefined
  }
})
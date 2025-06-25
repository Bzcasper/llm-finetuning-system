# PropertyGlow Environment Validation System

This document describes the comprehensive environment variable validation system implemented for PropertyGlow, a real estate SaaS platform.

## Overview

The environment validation system ensures that all required configuration is properly set before the application starts, providing clear error messages and fail-fast behavior for invalid configurations.

## Key Features

- **Comprehensive Validation**: Validates all required environment variables using Zod schemas
- **Environment-Specific Rules**: Different validation rules for development, staging, and production
- **Fail-Fast Behavior**: Application stops with clear errors if critical configuration is missing
- **Type Safety**: Full TypeScript support with type-safe environment variable access
- **Health Check Endpoint**: Runtime API endpoint to check configuration status
- **Clear Error Messages**: Detailed error messages with actionable recommendations

## File Structure

```
lib/
├── env-validation.ts    # Zod schemas and validation logic
├── env-loader.ts       # Startup validation and environment loading
└── config.ts          # Type-safe configuration access

app/api/env-check/      # Health check API endpoint
├── route.ts

types/
└── env.d.ts           # TypeScript type definitions

.env.example           # Comprehensive environment variable reference
```

## Required Environment Variables

### Core Application
- `NODE_ENV`: Application environment (development/staging/production)

### Database
- `DATABASE_URL`: PostgreSQL connection string

### Authentication (NextAuth.js)
- `NEXTAUTH_SECRET`: Secure secret key (minimum 32 characters)
- `NEXTAUTH_URL`: Application URL
- `GOOGLE_CLIENT_ID`: Google OAuth client ID
- `GOOGLE_CLIENT_SECRET`: Google OAuth client secret

### AI Services
- `FAL_KEY`: FAL.ai API key for image processing

### File Storage
- `BLOB_READ_WRITE_TOKEN`: Vercel Blob storage token

### Payment Processing (Stripe)
- `STRIPE_SECRET_KEY`: Stripe secret key (environment-specific prefix)
- `STRIPE_PUBLISHABLE_KEY`: Stripe publishable key (environment-specific prefix)
- `STRIPE_WEBHOOK_SECRET`: Stripe webhook secret

### Optional Services
- `SENTRY_DSN`: Sentry error monitoring (required in production)
- `ANALYTICS_ID`: Analytics tracking ID
- `MAPBOX_API_KEY`: Mapbox API key for mapping features

## Environment-Specific Validation

### Development
- Uses `sk_test_` and `pk_test_` prefixes for Stripe keys
- Optional variables can be omitted
- Relaxed validation rules

### Staging
- Uses `sk_test_` and `pk_test_` prefixes for Stripe keys
- Stricter validation than development
- Should mirror production setup

### Production
- Requires `sk_live_` and `pk_live_` prefixes for Stripe keys
- `SENTRY_DSN` is required for error monitoring
- Strictest validation rules
- All security configurations must be production-ready

## Usage

### Basic Usage

```typescript
import { config } from '@/lib/config'

// Type-safe access to environment variables
const databaseUrl = config.database.url
const stripeKey = config.payments.stripe.secretKey
const aiKey = config.ai.fal.key
```

### Manual Validation

```typescript
import { validateEnvironment } from '@/lib/env-validation'

try {
  const env = validateEnvironment()
  console.log('Environment validation successful')
} catch (error) {
  console.error('Environment validation failed:', error.message)
}
```

### Service-Specific Validation

```typescript
import { validateEnvGroup } from '@/lib/env-validation'

// Validate specific service groups
validateEnvGroup.auth()      // Authentication configuration
validateEnvGroup.database()  // Database configuration
validateEnvGroup.payments()  // Payment processing configuration
validateEnvGroup.ai()        // AI services configuration
validateEnvGroup.storage()   // File storage configuration
```

### Health Check API

Check environment status at runtime:

```bash
GET /api/env-check
```

Response includes:
- Overall environment health status
- Service-specific configuration status
- Detailed error messages and recommendations
- Debugging information

### Environment Loading

The environment is automatically loaded at application startup:

```typescript
import { loadEnvironment } from '@/lib/env-loader'

// Load with custom options
await loadEnvironment(process.env, {
  failFast: true,        // Fail immediately on validation errors
  logLevel: 'info'       // Logging level: silent, error, info, debug
})
```

## Error Handling

### Validation Errors

The system provides detailed error messages:

```
Environment validation failed for development environment:
  • DATABASE_URL: Required
  • NEXTAUTH_SECRET: Required
  • STRIPE_SECRET_KEY: Must start with 'sk_test_' in development
```

### Recommendations

The system provides actionable recommendations:

```
💡 To fix this issue, set the following environment variables:
   export DATABASE_URL="your-value-here"
   export NEXTAUTH_SECRET="your-value-here"

📝 Check your .env.example file for reference values
```

## TypeScript Support

Full TypeScript support with:
- Global `process.env` type declarations
- Type-safe configuration interfaces
- Environment validation result types
- Service status types

```typescript
// Types are automatically available
const dbUrl: string = process.env.DATABASE_URL
const nodeEnv: "development" | "staging" | "production" = process.env.NODE_ENV
```

## Testing

### Environment Validation Testing

The system includes comprehensive validation logic that can be tested with different environment configurations:

1. **Valid Development Environment**: All required variables with correct development prefixes
2. **Missing Required Variables**: Tests error handling for missing critical variables
3. **Invalid Stripe Configuration**: Tests environment-specific validation rules
4. **Valid Production Environment**: Tests production-specific requirements

### Health Check Testing

Test the health check endpoint:

```bash
curl http://localhost:3000/api/env-check
```

## Best Practices

### Development
1. Copy `.env.example` to `.env`
2. Fill in all required variables
3. Use test/sandbox keys for external services
4. Set `NODE_ENV=development`

### Staging
1. Use test/sandbox keys but with production-like values
2. Test all integrations thoroughly
3. Set `NODE_ENV=staging`
4. Monitor for any validation warnings

### Production
1. Use live keys for all services
2. Set `NODE_ENV=production`
3. Ensure `SENTRY_DSN` is configured for error monitoring
4. Use cryptographically secure secrets
5. Regularly rotate secrets and keys

### Security
1. Never commit `.env` files to version control
2. Use strong, unique secrets for `NEXTAUTH_SECRET`
3. Regularly rotate API keys and secrets
4. Use environment-specific prefixes for Stripe keys
5. Enable monitoring in production

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**
   - Check `.env` file exists and has correct variables
   - Compare with `.env.example`
   - Ensure no typos in variable names

2. **Invalid Stripe Keys**
   - Development: Use `sk_test_` and `pk_test_` prefixes
   - Production: Use `sk_live_` and `pk_live_` prefixes

3. **Database Connection Issues**
   - Verify `DATABASE_URL` format
   - Check database server is running
   - Validate connection string parameters

4. **Authentication Problems**
   - Ensure `NEXTAUTH_SECRET` is at least 32 characters
   - Verify Google OAuth credentials are correct
   - Check `NEXTAUTH_URL` matches your domain

### Debug Mode

Enable debug logging:

```typescript
await loadEnvironment(process.env, {
  logLevel: 'debug'
})
```

This will show detailed environment status for all services.

## Migration Guide

If upgrading from the old validation system:

1. Install the new validation system files
2. Update imports from `@/lib/config` to use new functions
3. Replace manual validation with the new system
4. Update `.env` files to include all required variables
5. Test with the health check endpoint

The old `validateEnvVars()` function is maintained for backward compatibility but is deprecated.

## Support

For issues with the environment validation system:
1. Check the health check endpoint: `/api/env-check`
2. Review server logs for detailed error messages  
3. Verify all required variables are set in `.env`
4. Compare your configuration with `.env.example`
5. Test individual service configurations using `validateEnvGroup`
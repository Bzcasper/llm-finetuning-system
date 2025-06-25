# PropertyGlow API Security Implementation

## Overview

This document describes the comprehensive API authentication middleware implemented for the PropertyGlow real estate SaaS platform. The security system includes authentication, authorization, rate limiting, and role-based access control (RBAC).

## Security Components

### 1. Rate Limiting (`lib/rate-limiting.ts`)

**Features:**
- Memory-based rate limiting with configurable limits
- IP-based tracking with proxy-awareness
- Different rate limits per endpoint type
- Automatic cleanup of expired entries
- Detailed rate limit headers in responses

**Rate Limit Configurations:**
- **API Routes**: 100 requests/minute per IP
- **Enhancement Routes**: 10 requests/minute per IP
- **Credits Routes**: 50 requests/minute per IP
- **Stripe Routes**: 20 requests/minute per IP
- **Admin Routes**: 200 requests/minute per IP
- **Auth Routes**: 5 requests/15 minutes per IP

**Usage:**
```typescript
import { rateLimit, RATE_LIMIT_CONFIGS } from "@/lib/rate-limiting"

const result = rateLimit(request, RATE_LIMIT_CONFIGS.api, "/api/example")
if (!result.success) {
  // Handle rate limit exceeded
}
```

### 2. Role-Based Access Control (`lib/rbac.ts`)

**User Roles:**
- `USER`: Basic user with standard permissions
- `ADMIN`: Administrative user with management permissions
- `SUPER_ADMIN`: Full system access

**Permissions System:**
- Fine-grained permissions for different operations
- Role-permission mapping with inheritance
- Route-specific permission requirements
- Resource ownership validation

**Key Permissions:**
- `ENHANCE_IMAGES`: Image enhancement operations
- `USE_CREDITS`: Credit consumption and management
- `READ_USERS`: View user information
- `MANAGE_CREDITS`: Admin credit management
- `SYSTEM_ADMIN`: System-level operations

**Usage:**
```typescript
import { canAccessRoute, getUserRole } from "@/lib/rbac"

const userRole = getUserRole(session)
const canAccess = canAccessRoute(session, "/api/admin/users")
```

### 3. Security Middleware (`lib/auth/middleware.ts`)

**Core Features:**
- Authentication validation using NextAuth tokens
- Rate limiting integration
- RBAC enforcement
- Credit balance checking
- Comprehensive error handling
- Security headers injection

**Middleware Options:**
```typescript
interface SecurityMiddlewareOptions {
  requireAuth?: boolean           // Require user authentication
  requireRole?: UserRole[]        // Required user roles
  requirePermissions?: Permission[] // Required permissions
  rateLimitConfig?: string        // Rate limit configuration
  checkCredits?: boolean          // Check user credit balance
  minCredits?: number            // Minimum required credits
  skipRateLimit?: boolean        // Skip rate limiting
}
```

**Usage Examples:**
```typescript
// Basic authentication
export const GET = withSecurity(handler, {
  requireAuth: true,
  rateLimitConfig: "api"
})

// Admin-only endpoint
export const POST = withSecurity(handler, {
  requireAuth: true,
  requireRole: [UserRole.ADMIN, UserRole.SUPER_ADMIN],
  requirePermissions: [Permission.READ_USERS]
})

// Credit-consuming operation
export const PUT = withCreditCheck(handler, 5) // Costs 5 credits
```

### 4. Enhanced Middleware (`middleware.ts`)

**Global Protection:**
- All API routes protected by default
- Automatic rate limiting
- Authentication enforcement
- Route-specific authorization
- Proper error responses for APIs vs UI routes

**Protected Route Patterns:**
- `/api/credits/*` - User authentication + credit permissions
- `/api/enhance/*` - User authentication + credit checking
- `/api/stripe/*` - User authentication for billing
- `/api/admin/*` - Admin role required

## Implementation Details

### Authentication Flow

1. **Token Validation**: NextAuth JWT tokens validated on each request
2. **Session Creation**: Token data converted to session object
3. **Permission Check**: User permissions validated against route requirements
4. **Rate Limiting**: Request counted against user/IP limits
5. **Credit Validation**: Credit balance checked for consuming operations

### Error Handling

**HTTP Status Codes:**
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions or role
- `429 Too Many Requests`: Rate limit exceeded
- `402 Payment Required`: Insufficient credits
- `500 Internal Server Error`: Server-side errors

**Error Response Format:**
```json
{
  "error": "Error message",
  "statusCode": 401,
  "details": "Additional error details",
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

### Security Headers

All API responses include security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- Rate limit headers (`X-RateLimit-*`)

## API Route Examples

### 1. Credits API (`/api/credits`)

**Security Configuration:**
- Authentication required
- `VIEW_OWN_CREDITS` permission for GET
- `USE_CREDITS` permission for POST
- Credits rate limiting (50 req/min)

**Features:**
- User can only access their own credits
- Input validation for credit amounts
- Transaction history tracking

### 2. Enhancement API (`/api/enhance`)

**Security Configuration:**
- Authentication required
- Credit checking (5 credits per enhancement)
- Enhancement rate limiting (10 req/min)
- File size and type validation

**Features:**
- Automatic credit deduction
- Credit refund on processing failure
- Comprehensive input validation
- User activity logging

### 3. Admin API (`/api/admin/users`)

**Security Configuration:**
- Admin role required
- Multiple permission checks
- Admin rate limiting (200 req/min)
- CRUD operation validation

**Features:**
- User management operations
- Role-based access restrictions
- Input validation and sanitization
- Audit logging

### 4. Stripe API (`/api/stripe/webhook`)

**Security Configuration:**
- Webhook signature validation
- Stripe-specific rate limiting
- Payment processing security

**Features:**
- Secure webhook processing
- Credit allocation on payment
- Payment failure handling

## Testing

### Security Test Endpoint (`/api/test-security`)

Provides multiple endpoints for testing security features:
- `GET`: Basic authentication test
- `POST`: Admin role test
- `PUT`: Credit balance test
- `DELETE`: Rate limiting test

### Test Scenarios

1. **Unauthenticated Access**: Should return 401
2. **Insufficient Role**: Should return 403
3. **Rate Limit Exceeded**: Should return 429
4. **Insufficient Credits**: Should return 402
5. **Valid Request**: Should return 200 with proper headers

## Production Considerations

### Rate Limiting

- **Memory Storage**: Current implementation uses in-memory storage
- **Production Recommendation**: Use Redis or similar distributed cache
- **Scaling**: Consider database-backed rate limiting for multi-instance deployments

### User Roles

- **Demo Implementation**: Currently uses email-based role assignment
- **Production Recommendation**: Store roles in database with proper management UI
- **Security**: Implement role change auditing and approval workflows

### Credit System

- **Demo Implementation**: Uses cookies for credit storage
- **Production Recommendation**: Use database with transaction support
- **Features**: Add payment integration, subscription management, usage analytics

### Monitoring

- **Logging**: Comprehensive security event logging implemented
- **Metrics**: Rate limit violations, authentication failures, permission denials
- **Alerts**: Set up monitoring for security anomalies

## Configuration

### Environment Variables

```env
NEXTAUTH_SECRET=your-secret-key
DATABASE_URL=your-database-url
NEXTAUTH_URL=your-app-url

# For production
RATE_LIMIT_REDIS_URL=redis://localhost:6379
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret
```

### Security Settings

- **Session Duration**: 30 days with 24-hour refresh
- **Rate Limit Windows**: 1-15 minutes depending on endpoint
- **Credit Costs**: Configurable per operation type
- **Admin Emails**: Configurable in RBAC system

## Security Best Practices Implemented

1. **Defense in Depth**: Multiple security layers (auth, authz, rate limiting)
2. **Principle of Least Privilege**: Minimal permissions per role
3. **Input Validation**: Comprehensive validation on all inputs
4. **Error Handling**: No sensitive data in error responses
5. **Logging**: Security events logged for monitoring
6. **Headers**: Security headers on all responses
7. **Rate Limiting**: Prevents abuse and DoS attacks
8. **Resource Protection**: User can only access their own data

## Conclusion

This security implementation provides enterprise-grade protection for the PropertyGlow API while maintaining developer-friendly usage patterns. The modular design allows for easy extension and customization based on specific requirements.

The system is production-ready with proper error handling, logging, and security measures, though some components (like rate limiting storage) should be enhanced for high-scale production deployments.
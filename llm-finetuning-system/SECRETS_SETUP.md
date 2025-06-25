# 🔐 GitHub Secrets Setup Guide for LLM Fine-Tuning Studio

This guide will help you configure all the necessary secrets for your GitHub repository to enable the CI/CD workflow.

## 📍 How to Add Secrets to GitHub

1. Go to your repository: https://github.com/Bzcasper/llm-finetuning-system
2. Click on **Settings** tab
3. In the left sidebar, click **Secrets and variables** → **Actions**
4. Click **New repository secret** for each secret below

## 🎯 Required Secrets

### 1. Modal.com Secrets (Required for AI Model Training)

**MODAL_TOKEN_ID**

```
Description: Modal.com API Token ID for GPU-based model training
How to get:
1. Go to https://modal.com
2. Sign up/Login to your account
3. Go to Settings → API Tokens
4. Create a new token
5. Copy the Token ID
Value: [Your Modal Token ID]
```

**MODAL_TOKEN_SECRET**

```
Description: Modal.com API Token Secret
How to get:
1. From the same Modal.com API token creation
2. Copy the Token Secret (keep this secure!)
Value: [Your Modal Token Secret]
```

### 2. Vercel Deployment Secrets (Required for Frontend Hosting)

**VERCEL_TOKEN**

```
Description: Vercel API token for deployment automation
How to get:
1. Go to https://vercel.com
2. Sign up/Login to your account
3. Go to Settings → Tokens
4. Create a new token with "Full Access"
5. Copy the token
Value: [Your Vercel Token]
```

**ORG_ID**

```
Description: Vercel Organization ID
How to get:
1. In Vercel dashboard, go to Settings → General
2. Copy your Team/User ID
3. Or run: vercel link (if you have Vercel CLI)
Value: [Your Vercel Org ID]
```

**PROJECT_ID**

```
Description: Vercel Project ID for your LLM Fine-tuning Studio
How to get:
1. Create a project on Vercel for this repository
2. Go to Project Settings → General
3. Copy the Project ID
4. Or run: vercel link (if you have Vercel CLI)
Value: [Your Vercel Project ID]
```

### 3. Database Secrets (Required for User Data)

**DATABASE_URL**

```
Description: PostgreSQL connection string for production database
Format: postgresql://username:password@host:port/database
Recommended: Use Neon, Supabase, or Railway for managed PostgreSQL

Neon (Recommended):
1. Go to https://neon.tech
2. Create a free account and database
3. Copy the connection string
4. Format: postgresql://[user]:[password]@[host]/[dbname]?sslmode=require

Value: [Your Database URL]
```

### 4. Authentication Secrets (Required for User Login)

**NEXTAUTH_SECRET**

```
Description: Secret key for NextAuth.js session encryption
How to generate: Use a random 32-character string
Command to generate: openssl rand -base64 32
Value: [Your NextAuth Secret - 32 random characters]
```

**NEXTAUTH_URL**

```
Description: Base URL for NextAuth callbacks
For Production: https://llm-finetuning-studio.vercel.app
For Staging: https://llm-finetuning-studio-staging.vercel.app
Value: [Your Production URL]
```

**GOOGLE_CLIENT_ID**

```
Description: Google OAuth Client ID for Google Sign-in
How to get:
1. Go to https://console.cloud.google.com
2. Create a new project or select existing
3. Enable Google+ API
4. Go to Credentials → Create OAuth 2.0 Client
5. Add your domain to authorized origins
6. Copy Client ID
Value: [Your Google Client ID]
```

**GOOGLE_CLIENT_SECRET**

```
Description: Google OAuth Client Secret
How to get: From the same Google OAuth setup above
Value: [Your Google Client Secret]
```

**GITHUB_ID**

```
Description: GitHub OAuth App ID for GitHub Sign-in
How to get:
1. Go to GitHub Settings → Developer settings → OAuth Apps
2. Create a new OAuth App
3. Authorization callback URL: [your-domain]/api/auth/callback/github
4. Copy Client ID
Value: [Your GitHub OAuth Client ID]
```

**GITHUB_SECRET**

```
Description: GitHub OAuth App Secret
How to get: From the same GitHub OAuth App setup above
Value: [Your GitHub OAuth Client Secret]
```

### 5. Payment Processing (Required for Subscriptions)

**STRIPE_SECRET_KEY**

```
Description: Stripe secret key for payment processing
How to get:
1. Go to https://stripe.com
2. Create account and get verified
3. Go to Developers → API Keys
4. Copy the Secret key (starts with sk_live_ or sk_test_)
Value: [Your Stripe Secret Key]
```

**STRIPE_PUBLISHABLE_KEY**

```
Description: Stripe publishable key for frontend
How to get: From the same Stripe API Keys page
Value: [Your Stripe Publishable Key]
```

**STRIPE_WEBHOOK_SECRET**

```
Description: Stripe webhook endpoint secret for event verification
How to get:
1. In Stripe Dashboard → Webhooks
2. Create endpoint: [your-domain]/api/stripe/webhook
3. Select events: checkout.session.completed, customer.subscription.updated
4. Copy the webhook secret
Value: [Your Stripe Webhook Secret]
```

### 6. Email Service (Required for Notifications)

**RESEND_API_KEY**

```
Description: Resend API key for transactional emails
How to get:
1. Go to https://resend.com
2. Create account and verify domain
3. Go to API Keys and create new key
Value: [Your Resend API Key]
```

**RESEND_FROM_EMAIL**

```
Description: Verified sender email address
Format: noreply@yourdomain.com
Note: Must be verified in Resend dashboard
Value: [Your verified sender email]
```

### 7. File Storage (Required for Dataset Management)

**MINIO_ENDPOINT**

```
Description: MinIO/S3 endpoint for file storage
Options:
- Self-hosted MinIO: your-minio-domain.com
- AWS S3: s3.amazonaws.com
- DigitalOcean Spaces: nyc3.digitaloceanspaces.com
Value: [Your storage endpoint]
```

**MINIO_ACCESS_KEY**

```
Description: MinIO/S3 access key
How to get: From your MinIO admin panel or AWS IAM
Value: [Your access key]
```

**MINIO_SECRET_KEY**

```
Description: MinIO/S3 secret key
How to get: From your MinIO admin panel or AWS IAM
Value: [Your secret key]
```

**MINIO_SECURE**

```
Description: Use HTTPS for MinIO connections
Value: true (for production) or false (for local development)
```

### 8. Notifications (Optional)

**SLACK_WEBHOOK**

```
Description: Slack webhook URL for deployment notifications
How to get:
1. Go to your Slack workspace
2. Apps → Incoming Webhooks
3. Create webhook for #deployments channel
4. Copy webhook URL
Value: [Your Slack webhook URL]
```

## 🚀 Quick Setup Commands

After setting up all secrets, test the workflow:

```bash
# Trigger workflow manually
gh workflow run ci-cd.yml

# Check workflow status
gh run list

# View workflow logs
gh run view [run-id] --log
```

## 🔍 Verification Checklist

- [ ] MODAL_TOKEN_ID and MODAL_TOKEN_SECRET (Modal.com)
- [ ] VERCEL_TOKEN, ORG_ID, PROJECT_ID (Vercel)
- [ ] DATABASE_URL (PostgreSQL)
- [ ] NEXTAUTH_SECRET, NEXTAUTH_URL (Authentication)
- [ ] GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET (Google OAuth)
- [ ] GITHUB_ID, GITHUB_SECRET (GitHub OAuth)
- [ ] STRIPE_SECRET_KEY, STRIPE_PUBLISHABLE_KEY, STRIPE_WEBHOOK_SECRET (Payments)
- [ ] RESEND_API_KEY, RESEND_FROM_EMAIL (Email)
- [ ] MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE (Storage)
- [ ] SLACK_WEBHOOK (Optional notifications)

## 🛡️ Security Best Practices

1. **Never commit secrets** to your repository
2. **Use environment-specific secrets** for staging vs production
3. **Rotate secrets regularly** (every 90 days)
4. **Use least-privilege access** for all API keys
5. **Monitor secret usage** in service dashboards

## 🆘 Troubleshooting

**Workflow fails with authentication errors:**

- Double-check all token formats and permissions
- Ensure tokens haven't expired
- Verify API quotas aren't exceeded

**Deployment fails:**

- Check Vercel project settings
- Verify all environment variables are set
- Review build logs for missing dependencies

**Modal deployment fails:**

- Confirm Modal tokens have GPU access
- Check Modal account billing and limits
- Verify Modal CLI authentication

---

**Need Help?** Create an issue in the repository with your error logs (remove any sensitive information).

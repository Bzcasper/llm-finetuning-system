#!/bin/bash

# GitHub Secrets Setup Script for LLM Fine-Tuning Studio
# This script will prompt you for each secret and set them in your GitHub repository

echo "🔐 GitHub Secrets Setup for LLM Fine-Tuning Studio"
echo "=================================================="
echo ""
echo "This script will help you set up all required secrets for your GitHub workflow."
echo "Press Ctrl+C at any time to exit."
echo ""

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed. Please install it first:"
    echo "   https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ You are not authenticated with GitHub CLI."
    echo "   Run: gh auth login"
    exit 1
fi

echo "✅ GitHub CLI is ready!"
echo ""

# Function to set secret
set_secret() {
    local secret_name=$1
    local description=$2
    local example=$3
    
    echo "🔑 Setting up: $secret_name"
    echo "Description: $description"
    if [ ! -z "$example" ]; then
        echo "Example: $example"
    fi
    echo ""
    
    read -p "Enter value for $secret_name (or press Enter to skip): " secret_value
    
    if [ ! -z "$secret_value" ]; then
        echo "$secret_value" | gh secret set "$secret_name"
        if [ $? -eq 0 ]; then
            echo "✅ $secret_name set successfully!"
        else
            echo "❌ Failed to set $secret_name"
        fi
    else
        echo "⏭️  Skipped $secret_name"
    fi
    echo ""
}

# Function to set optional secret
set_optional_secret() {
    local secret_name=$1
    local description=$2
    local example=$3
    
    read -p "Do you want to set up $secret_name? (y/N): " setup_optional
    if [[ $setup_optional =~ ^[Yy]$ ]]; then
        set_secret "$secret_name" "$description" "$example"
    else
        echo "⏭️  Skipped optional secret: $secret_name"
        echo ""
    fi
}

echo "📋 REQUIRED SECRETS"
echo "==================="

# Modal.com secrets
echo "🚀 1. Modal.com (AI Training Platform)"
set_secret "MODAL_TOKEN_ID" "Modal.com API Token ID for GPU training" "tok_abc123..."
set_secret "MODAL_TOKEN_SECRET" "Modal.com API Token Secret" "secret_xyz789..."

# Vercel secrets
echo "🌐 2. Vercel (Frontend Hosting)"
set_secret "VERCEL_TOKEN" "Vercel API token for deployment" "vercel_token_..."
set_secret "ORG_ID" "Vercel Organization ID" "team_abc123..."
set_secret "PROJECT_ID" "Vercel Project ID" "prj_abc123..."

# Database
echo "🗄️ 3. Database"
set_secret "DATABASE_URL" "PostgreSQL connection string" "postgresql://user:pass@host:5432/db"

# Authentication
echo "🔐 4. Authentication"
set_secret "NEXTAUTH_SECRET" "NextAuth.js secret (32 random chars)" "$(openssl rand -base64 32 2>/dev/null || echo 'generate-32-char-secret')"
set_secret "NEXTAUTH_URL" "Your production URL" "https://your-domain.vercel.app"

# OAuth providers
echo "🔑 5. OAuth Providers"
set_secret "GOOGLE_CLIENT_ID" "Google OAuth Client ID" "123456789-abc.apps.googleusercontent.com"
set_secret "GOOGLE_CLIENT_SECRET" "Google OAuth Client Secret" "GOCSPX-abc123..."
set_secret "GITHUB_ID" "GitHub OAuth App Client ID" "Iv1.abc123..."
set_secret "GITHUB_SECRET" "GitHub OAuth App Secret" "abc123..."

# Stripe
echo "💳 6. Stripe (Payment Processing)"
set_secret "STRIPE_SECRET_KEY" "Stripe secret key" "sk_live_... or sk_test_..."
set_secret "STRIPE_PUBLISHABLE_KEY" "Stripe publishable key" "pk_live_... or pk_test_..."
set_secret "STRIPE_WEBHOOK_SECRET" "Stripe webhook secret" "whsec_..."

# Email service
echo "📧 7. Email Service"
set_secret "RESEND_API_KEY" "Resend API key for emails" "re_..."
set_secret "RESEND_FROM_EMAIL" "Verified sender email" "noreply@yourdomain.com"

# File storage
echo "📁 8. File Storage (MinIO/S3)"
set_secret "MINIO_ENDPOINT" "Storage endpoint" "s3.amazonaws.com or your-minio.com"
set_secret "MINIO_ACCESS_KEY" "Storage access key" "AKIA..."
set_secret "MINIO_SECRET_KEY" "Storage secret key" "secret123..."
set_secret "MINIO_SECURE" "Use HTTPS for storage" "true"

echo ""
echo "📋 OPTIONAL SECRETS"
echo "==================="

# Slack notifications
set_optional_secret "SLACK_WEBHOOK" "Slack webhook for notifications" "https://hooks.slack.com/services/..."

echo ""
echo "🎉 Secret setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Verify all secrets in GitHub: https://github.com/$(gh repo view --json owner,name -q '.owner.login + \"/\" + .name')/settings/secrets/actions"
echo "2. Test the workflow: gh workflow run ci-cd.yml"
echo "3. Monitor the run: gh run list"
echo ""
echo "📚 For detailed setup instructions, see: SECRETS_SETUP.md"
echo ""

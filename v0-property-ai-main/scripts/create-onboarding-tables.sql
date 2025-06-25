-- Add onboarding fields to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS company_name VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS user_type VARCHAR(100);
ALTER TABLE users ADD COLUMN IF NOT EXISTS monthly_volume VARCHAR(50);
ALTER TABLE users ADD COLUMN IF NOT EXISTS discovery_source VARCHAR(100);
ALTER TABLE users ADD COLUMN IF NOT EXISTS onboarding_completed BOOLEAN DEFAULT FALSE;

-- Create user preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
  id SERIAL PRIMARY KEY,
  user_email VARCHAR(255) REFERENCES users(email) ON DELETE CASCADE,
  initial_credits INTEGER DEFAULT 25,
  preferred_enhancement_type VARCHAR(100) DEFAULT 'general_enhancement',
  notification_preferences JSONB DEFAULT '{"email": true, "push": false}',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(user_email)
);

-- Create onboarding analytics table
CREATE TABLE IF NOT EXISTS onboarding_analytics (
  id SERIAL PRIMARY KEY,
  user_email VARCHAR(255) REFERENCES users(email) ON DELETE CASCADE,
  completion_time INTERVAL,
  user_agent TEXT,
  referrer TEXT,
  completed_at TIMESTAMP DEFAULT NOW()
);

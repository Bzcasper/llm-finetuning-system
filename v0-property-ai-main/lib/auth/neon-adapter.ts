import type { Adapter } from "next-auth/adapters"
import type { NeonQueryFunction } from "@neondatabase/serverless"
import { randomUUID } from "crypto"

export function NeonAdapter(sql: NeonQueryFunction): Adapter {
  return {
    async createUser(user) {
      const userId = randomUUID()
      const result = await sql`
        INSERT INTO users (id, name, email, email_verified, image)
        VALUES (${userId}, ${user.name}, ${user.email}, ${user.emailVerified?.toISOString() || null}, ${user.image})
        RETURNING id, name, email, email_verified, image
      `
      const dbUser = result[0]
      
      // Create corresponding user profile
      await sql`
        INSERT INTO user_profiles (id, user_id)
        VALUES (${randomUUID()}, ${userId})
      `
      
      return {
        id: dbUser.id,
        name: dbUser.name,
        email: dbUser.email,
        emailVerified: dbUser.email_verified ? new Date(dbUser.email_verified) : null,
        image: dbUser.image,
      }
    },

    async getUser(id) {
      const result = await sql`
        SELECT id, name, email, email_verified, image
        FROM users
        WHERE id = ${id}
      `
      if (result.length === 0) return null

      const user = result[0]
      return {
        id: user.id,
        name: user.name,
        email: user.email,
        emailVerified: user.email_verified ? new Date(user.email_verified) : null,
        image: user.image,
      }
    },

    async getUserByEmail(email) {
      const result = await sql`
        SELECT id, name, email, email_verified, image
        FROM users
        WHERE email = ${email}
      `
      if (result.length === 0) return null

      const user = result[0]
      return {
        id: user.id,
        name: user.name,
        email: user.email,
        emailVerified: user.email_verified ? new Date(user.email_verified) : null,
        image: user.image,
      }
    },

    async getUserByAccount({ providerAccountId, provider }) {
      const result = await sql`
        SELECT u.id, u.name, u.email, u.email_verified, u.image
        FROM users u
        JOIN accounts a ON u.id = a.user_id
        WHERE a.provider = ${provider}
        AND a.provider_account_id = ${providerAccountId}
      `
      if (result.length === 0) return null

      const user = result[0]
      return {
        id: user.id,
        name: user.name,
        email: user.email,
        emailVerified: user.email_verified ? new Date(user.email_verified) : null,
        image: user.image,
      }
    },

    async updateUser(user) {
      const result = await sql`
        UPDATE users
        SET name = ${user.name}, email = ${user.email}, image = ${user.image}
        WHERE id = ${user.id}
        RETURNING id, name, email, email_verified, image
      `
      const dbUser = result[0]
      return {
        id: dbUser.id,
        name: dbUser.name,
        email: dbUser.email,
        emailVerified: dbUser.email_verified ? new Date(dbUser.email_verified) : null,
        image: dbUser.image,
      }
    },

    async linkAccount(account) {
      const accountId = randomUUID()
      await sql`
        INSERT INTO accounts (
          id, user_id, provider, type, provider_account_id, 
          refresh_token, access_token, expires_at, token_type, scope, id_token
        )
        VALUES (
          ${accountId}, ${account.userId}, ${account.provider}, ${account.type}, ${account.providerAccountId},
          ${account.refresh_token}, ${account.access_token}, ${account.expires_at},
          ${account.token_type}, ${account.scope}, ${account.id_token}
        )
      `
      return { ...account, id: accountId }
    },

    async createSession(session) {
      const sessionId = randomUUID()
      await sql`
        INSERT INTO sessions (id, user_id, expires, session_token)
        VALUES (${sessionId}, ${session.userId}, ${new Date(session.expires).toISOString()}, ${session.sessionToken})
      `
      return { ...session, id: sessionId }
    },

    async getSessionAndUser(sessionToken) {
      const result = await sql`
        SELECT s.user_id, s.expires, s.session_token,
               u.id, u.name, u.email, u.email_verified, u.image
        FROM sessions s
        JOIN users u ON s.user_id = u.id
        WHERE s.session_token = ${sessionToken}
      `
      if (result.length === 0) return null

      const session = result[0]
      return {
        session: {
          userId: session.user_id,
          expires: new Date(session.expires),
          sessionToken: session.session_token,
        },
        user: {
          id: session.id,
          name: session.name,
          email: session.email,
          emailVerified: session.email_verified ? new Date(session.email_verified) : null,
          image: session.image,
        },
      }
    },

    async updateSession(session) {
      const result = await sql`
        UPDATE sessions
        SET expires = ${new Date(session.expires).toISOString()}
        WHERE session_token = ${session.sessionToken}
        RETURNING user_id, expires, session_token
      `
      if (result.length === 0) return null

      const dbSession = result[0]
      return {
        userId: dbSession.user_id,
        expires: new Date(dbSession.expires),
        sessionToken: dbSession.session_token,
      }
    },

    async deleteSession(sessionToken) {
      await sql`
        DELETE FROM sessions
        WHERE session_token = ${sessionToken}
      `
    },

    async createVerificationToken(verificationToken) {
      await sql`
        INSERT INTO verification_tokens (identifier, token, expires)
        VALUES (${verificationToken.identifier}, ${verificationToken.token}, ${new Date(verificationToken.expires).toISOString()})
      `
      return verificationToken
    },

    async unlinkAccount({ providerAccountId, provider }) {
      await sql`
        DELETE FROM accounts 
        WHERE provider = ${provider} AND provider_account_id = ${providerAccountId}
      `
    },

    async deleteUser(userId) {
      await sql`DELETE FROM users WHERE id = ${userId}`
    },

    async useVerificationToken({ identifier, token }) {
      const result = await sql`
        DELETE FROM verification_tokens
        WHERE identifier = ${identifier} AND token = ${token}
        RETURNING identifier, token, expires
      `
      if (result.length === 0) return null

      const verificationToken = result[0]
      return {
        identifier: verificationToken.identifier,
        token: verificationToken.token,
        expires: new Date(verificationToken.expires),
      }
    },
  }
}

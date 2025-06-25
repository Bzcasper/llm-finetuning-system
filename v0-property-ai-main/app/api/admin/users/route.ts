/**
 * Admin API - User Management
 * Requires admin privileges to access
 */

import { type NextRequest, NextResponse } from "next/server"
import { withSecurity } from "@/lib/auth/middleware"
import { Permission, UserRole } from "@/lib/rbac"

export const GET = withSecurity(
  async (request: NextRequest) => {
    try {
      // In a real app, this would query the database for users
      const mockUsers = [
        {
          id: "1",
          name: "John Doe",
          email: "john@example.com",
          role: UserRole.USER,
          credits: 50,
          createdAt: "2024-01-01T00:00:00Z",
          lastActive: "2024-01-15T12:00:00Z",
        },
        {
          id: "2",
          name: "Jane Admin",
          email: "jane@propertyglow.com",
          role: UserRole.ADMIN,
          credits: 1000,
          createdAt: "2023-12-01T00:00:00Z",
          lastActive: "2024-01-15T14:30:00Z",
        },
      ]

      // Get pagination parameters
      const page = parseInt(request.nextUrl.searchParams.get("page") || "1")
      const limit = parseInt(request.nextUrl.searchParams.get("limit") || "10")
      const search = request.nextUrl.searchParams.get("search") || ""

      // Filter users based on search
      let filteredUsers = mockUsers
      if (search) {
        filteredUsers = mockUsers.filter(
          user => 
            user.name.toLowerCase().includes(search.toLowerCase()) ||
            user.email.toLowerCase().includes(search.toLowerCase())
        )
      }

      // Apply pagination
      const startIndex = (page - 1) * limit
      const endIndex = startIndex + limit
      const paginatedUsers = filteredUsers.slice(startIndex, endIndex)

      return NextResponse.json({
        success: true,
        users: paginatedUsers,
        pagination: {
          page,
          limit,
          total: filteredUsers.length,
          totalPages: Math.ceil(filteredUsers.length / limit),
        },
        timestamp: new Date().toISOString(),
      })
    } catch (error) {
      console.error("Error fetching users:", error)

      return NextResponse.json(
        {
          error: "Failed to fetch users",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      )
    }
  },
  {
    requireAuth: true,
    requireRole: [UserRole.ADMIN, UserRole.SUPER_ADMIN],
    requirePermissions: [Permission.READ_USERS],
    rateLimitConfig: "admin",
  }
)

export const POST = withSecurity(
  async (request: NextRequest) => {
    try {
      const body = await request.json()
      const { name, email, role = UserRole.USER, initialCredits = 10 } = body

      // Validate input
      if (!name || !email) {
        return NextResponse.json(
          {
            error: "Missing required fields",
            details: "Name and email are required",
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        )
      }

      // Validate email format
      const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
      if (!emailRegex.test(email)) {
        return NextResponse.json(
          {
            error: "Invalid email format",
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        )
      }

      // Validate role
      if (!Object.values(UserRole).includes(role)) {
        return NextResponse.json(
          {
            error: "Invalid role",
            details: `Role must be one of: ${Object.values(UserRole).join(", ")}`,
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        )
      }

      // In a real app, this would create the user in the database
      const newUser = {
        id: Math.random().toString(36).substring(2, 15),
        name,
        email,
        role,
        credits: initialCredits,
        createdAt: new Date().toISOString(),
        lastActive: null,
      }

      console.log("Admin: Created new user:", { id: newUser.id, email: newUser.email, role: newUser.role })

      return NextResponse.json({
        success: true,
        user: newUser,
        message: "User created successfully",
        timestamp: new Date().toISOString(),
      })
    } catch (error) {
      console.error("Error creating user:", error)

      return NextResponse.json(
        {
          error: "Failed to create user",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      )
    }
  },
  {
    requireAuth: true,
    requireRole: [UserRole.ADMIN, UserRole.SUPER_ADMIN],
    requirePermissions: [Permission.UPDATE_USERS],
    rateLimitConfig: "admin",
  }
)

export const PUT = withSecurity(
  async (request: NextRequest) => {
    try {
      const body = await request.json()
      const { userId, name, email, role, credits } = body

      // Validate input
      if (!userId) {
        return NextResponse.json(
          {
            error: "Missing user ID",
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        )
      }

      // In a real app, this would update the user in the database
      const updatedUser = {
        id: userId,
        name: name || "Updated User",
        email: email || "updated@example.com",
        role: role || UserRole.USER,
        credits: credits !== undefined ? credits : 50,
        updatedAt: new Date().toISOString(),
      }

      console.log("Admin: Updated user:", { id: updatedUser.id, changes: { name, email, role, credits } })

      return NextResponse.json({
        success: true,
        user: updatedUser,
        message: "User updated successfully",
        timestamp: new Date().toISOString(),
      })
    } catch (error) {
      console.error("Error updating user:", error)

      return NextResponse.json(
        {
          error: "Failed to update user",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      )
    }
  },
  {
    requireAuth: true,
    requireRole: [UserRole.ADMIN, UserRole.SUPER_ADMIN],
    requirePermissions: [Permission.UPDATE_USERS],
    rateLimitConfig: "admin",
  }
)

export const DELETE = withSecurity(
  async (request: NextRequest) => {
    try {
      const { searchParams } = new URL(request.url)
      const userId = searchParams.get("userId")

      if (!userId) {
        return NextResponse.json(
          {
            error: "Missing user ID",
            timestamp: new Date().toISOString(),
          },
          { status: 400 }
        )
      }

      // In a real app, this would delete the user from the database
      console.log("Admin: Deleted user:", { id: userId })

      return NextResponse.json({
        success: true,
        message: "User deleted successfully",
        deletedUserId: userId,
        timestamp: new Date().toISOString(),
      })
    } catch (error) {
      console.error("Error deleting user:", error)

      return NextResponse.json(
        {
          error: "Failed to delete user",
          details: error instanceof Error ? error.message : String(error),
          timestamp: new Date().toISOString(),
        },
        { status: 500 }
      )
    }
  },
  {
    requireAuth: true,
    requireRole: [UserRole.SUPER_ADMIN], // Only super admin can delete users
    requirePermissions: [Permission.DELETE_USERS],
    rateLimitConfig: "admin",
  }
)
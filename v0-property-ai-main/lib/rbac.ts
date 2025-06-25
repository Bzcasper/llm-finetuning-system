/**
 * Role-Based Access Control (RBAC) system for PropertyGlow
 * Defines user roles, permissions, and access control logic
 */

import { Session } from "next-auth"
import { NextRequest } from "next/server"

// Define user roles
export enum UserRole {
  USER = "user",
  ADMIN = "admin",
  SUPER_ADMIN = "super_admin",
}

// Define permissions
export enum Permission {
  // User permissions
  READ_OWN_PROFILE = "read:own_profile",
  UPDATE_OWN_PROFILE = "update:own_profile",
  USE_CREDITS = "use:credits",
  VIEW_OWN_CREDITS = "view:own_credits",
  ENHANCE_IMAGES = "enhance:images",
  VIEW_OWN_HISTORY = "view:own_history",
  
  // Admin permissions
  READ_USERS = "read:users",
  UPDATE_USERS = "update:users",
  VIEW_ALL_CREDITS = "view:all_credits",
  MANAGE_CREDITS = "manage:credits",
  VIEW_ANALYTICS = "view:analytics",
  MANAGE_SYSTEM_SETTINGS = "manage:system_settings",
  
  // Super admin permissions
  DELETE_USERS = "delete:users",
  MANAGE_ROLES = "manage:roles",
  SYSTEM_ADMIN = "system:admin",
  MANAGE_BILLING = "manage:billing",
  VIEW_LOGS = "view:logs",
}

// Role-Permission mapping
export const ROLE_PERMISSIONS: Record<UserRole, Permission[]> = {
  [UserRole.USER]: [
    Permission.READ_OWN_PROFILE,
    Permission.UPDATE_OWN_PROFILE,
    Permission.USE_CREDITS,
    Permission.VIEW_OWN_CREDITS,
    Permission.ENHANCE_IMAGES,
    Permission.VIEW_OWN_HISTORY,
  ],
  [UserRole.ADMIN]: [
    // Include all user permissions
    ...ROLE_PERMISSIONS[UserRole.USER] || [],
    Permission.READ_USERS,
    Permission.UPDATE_USERS,
    Permission.VIEW_ALL_CREDITS,
    Permission.MANAGE_CREDITS,
    Permission.VIEW_ANALYTICS,
    Permission.MANAGE_SYSTEM_SETTINGS,
  ],
  [UserRole.SUPER_ADMIN]: [
    // Include all admin permissions
    ...ROLE_PERMISSIONS[UserRole.ADMIN] || [],
    Permission.DELETE_USERS,
    Permission.MANAGE_ROLES,
    Permission.SYSTEM_ADMIN,
    Permission.MANAGE_BILLING,
    Permission.VIEW_LOGS,
  ],
}

// Fix circular reference issue
ROLE_PERMISSIONS[UserRole.ADMIN] = [
  Permission.READ_OWN_PROFILE,
  Permission.UPDATE_OWN_PROFILE,
  Permission.USE_CREDITS,
  Permission.VIEW_OWN_CREDITS,
  Permission.ENHANCE_IMAGES,
  Permission.VIEW_OWN_HISTORY,
  Permission.READ_USERS,
  Permission.UPDATE_USERS,
  Permission.VIEW_ALL_CREDITS,
  Permission.MANAGE_CREDITS,
  Permission.VIEW_ANALYTICS,
  Permission.MANAGE_SYSTEM_SETTINGS,
]

ROLE_PERMISSIONS[UserRole.SUPER_ADMIN] = [
  ...ROLE_PERMISSIONS[UserRole.ADMIN],
  Permission.DELETE_USERS,
  Permission.MANAGE_ROLES,
  Permission.SYSTEM_ADMIN,
  Permission.MANAGE_BILLING,
  Permission.VIEW_LOGS,
]

// Route-Permission mapping
export const ROUTE_PERMISSIONS: Record<string, Permission[]> = {
  // Credits API
  "/api/credits": [Permission.VIEW_OWN_CREDITS, Permission.USE_CREDITS],
  
  // Enhancement APIs
  "/api/enhance": [Permission.ENHANCE_IMAGES, Permission.USE_CREDITS],
  "/api/enhance-pro": [Permission.ENHANCE_IMAGES, Permission.USE_CREDITS],
  "/api/enhance-simple": [Permission.ENHANCE_IMAGES, Permission.USE_CREDITS],
  "/api/enhance-with-model": [Permission.ENHANCE_IMAGES, Permission.USE_CREDITS],
  
  // Stripe/Billing APIs
  "/api/stripe": [Permission.USE_CREDITS], // Users can manage their own billing
  
  // Admin APIs
  "/api/admin": [Permission.READ_USERS, Permission.MANAGE_CREDITS],
  "/api/admin/users": [Permission.READ_USERS, Permission.UPDATE_USERS],
  "/api/admin/credits": [Permission.VIEW_ALL_CREDITS, Permission.MANAGE_CREDITS],
  "/api/admin/analytics": [Permission.VIEW_ANALYTICS],
  "/api/admin/settings": [Permission.MANAGE_SYSTEM_SETTINGS],
  "/api/admin/logs": [Permission.VIEW_LOGS],
  
  // Super admin APIs
  "/api/admin/users/delete": [Permission.DELETE_USERS],
  "/api/admin/roles": [Permission.MANAGE_ROLES],
  "/api/admin/system": [Permission.SYSTEM_ADMIN],
}

/**
 * Get user role from session
 */
export function getUserRole(session: Session | null): UserRole {
  if (!session?.user) {
    return UserRole.USER // Default role for unauthenticated users (though they won't pass auth checks)
  }
  
  // Check if user has a role defined in the session
  // In a real application, this would come from the database
  const userEmail = session.user.email?.toLowerCase()
  
  // For demo purposes, define some admin users
  // In production, this should come from the database
  const adminEmails = [
    "admin@propertyglow.com",
    "support@propertyglow.com",
  ]
  
  const superAdminEmails = [
    "superadmin@propertyglow.com",
    "owner@propertyglow.com",
  ]
  
  if (userEmail && superAdminEmails.includes(userEmail)) {
    return UserRole.SUPER_ADMIN
  }
  
  if (userEmail && adminEmails.includes(userEmail)) {
    return UserRole.ADMIN
  }
  
  return UserRole.USER
}

/**
 * Get permissions for a role
 */
export function getRolePermissions(role: UserRole): Permission[] {
  return ROLE_PERMISSIONS[role] || []
}

/**
 * Check if a role has a specific permission
 */
export function hasPermission(role: UserRole, permission: Permission): boolean {
  const permissions = getRolePermissions(role)
  return permissions.includes(permission)
}

/**
 * Check if a user has a specific permission
 */
export function userHasPermission(session: Session | null, permission: Permission): boolean {
  if (!session) {
    return false
  }
  
  const role = getUserRole(session)
  return hasPermission(role, permission)
}

/**
 * Check if a user can access a specific route
 */
export function canAccessRoute(session: Session | null, routePath: string): boolean {
  if (!session) {
    return false
  }
  
  const role = getUserRole(session)
  const requiredPermissions = getRoutePermissions(routePath)
  
  // If no specific permissions required, allow access for authenticated users
  if (requiredPermissions.length === 0) {
    return true
  }
  
  // Check if user has any of the required permissions
  return requiredPermissions.some(permission => hasPermission(role, permission))
}

/**
 * Get required permissions for a route
 */
export function getRoutePermissions(routePath: string): Permission[] {
  // Check for exact match first
  if (ROUTE_PERMISSIONS[routePath]) {
    return ROUTE_PERMISSIONS[routePath]
  }
  
  // Check for pattern matches
  for (const [pattern, permissions] of Object.entries(ROUTE_PERMISSIONS)) {
    if (pattern.includes("*") || routePath.startsWith(pattern.replace("*", ""))) {
      return permissions
    }
  }
  
  // Default: require authentication but no specific permissions
  return []
}

/**
 * Check if a user has admin privileges
 */
export function isAdmin(session: Session | null): boolean {
  const role = getUserRole(session)
  return role === UserRole.ADMIN || role === UserRole.SUPER_ADMIN
}

/**
 * Check if a user has super admin privileges
 */
export function isSuperAdmin(session: Session | null): boolean {
  const role = getUserRole(session)
  return role === UserRole.SUPER_ADMIN
}

/**
 * Create RBAC middleware for API routes
 */
export function createRBACMiddleware(requiredPermissions: Permission[]) {
  return (session: Session | null) => {
    if (!session) {
      return {
        success: false,
        error: "Authentication required",
        statusCode: 401,
      }
    }
    
    const role = getUserRole(session)
    const hasRequiredPermission = requiredPermissions.some(permission => 
      hasPermission(role, permission)
    )
    
    if (!hasRequiredPermission) {
      return {
        success: false,
        error: "Insufficient permissions",
        statusCode: 403,
      }
    }
    
    return {
      success: true,
      role,
      permissions: getRolePermissions(role),
    }
  }
}

/**
 * Validate resource ownership (for user-specific resources)
 */
export function canAccessUserResource(
  session: Session | null,
  resourceUserId: string
): boolean {
  if (!session) {
    return false
  }
  
  const role = getUserRole(session)
  
  // Super admins and admins can access any resource
  if (role === UserRole.SUPER_ADMIN || role === UserRole.ADMIN) {
    return true
  }
  
  // Users can only access their own resources
  return session.user.id === resourceUserId
}

/**
 * Get allowed actions for a user on a specific resource type
 */
export function getAllowedActions(
  session: Session | null,
  resourceType: "user" | "credits" | "enhancement" | "admin"
): string[] {
  if (!session) {
    return []
  }
  
  const role = getUserRole(session)
  const permissions = getRolePermissions(role)
  
  const actions: string[] = []
  
  switch (resourceType) {
    case "user":
      if (permissions.includes(Permission.READ_OWN_PROFILE)) actions.push("read_own")
      if (permissions.includes(Permission.UPDATE_OWN_PROFILE)) actions.push("update_own")
      if (permissions.includes(Permission.READ_USERS)) actions.push("read_all")
      if (permissions.includes(Permission.UPDATE_USERS)) actions.push("update_all")
      if (permissions.includes(Permission.DELETE_USERS)) actions.push("delete")
      break
      
    case "credits":
      if (permissions.includes(Permission.VIEW_OWN_CREDITS)) actions.push("view_own")
      if (permissions.includes(Permission.USE_CREDITS)) actions.push("use")
      if (permissions.includes(Permission.VIEW_ALL_CREDITS)) actions.push("view_all")
      if (permissions.includes(Permission.MANAGE_CREDITS)) actions.push("manage")
      break
      
    case "enhancement":
      if (permissions.includes(Permission.ENHANCE_IMAGES)) actions.push("enhance")
      break
      
    case "admin":
      if (permissions.includes(Permission.VIEW_ANALYTICS)) actions.push("analytics")
      if (permissions.includes(Permission.MANAGE_SYSTEM_SETTINGS)) actions.push("settings")
      if (permissions.includes(Permission.VIEW_LOGS)) actions.push("logs")
      if (permissions.includes(Permission.SYSTEM_ADMIN)) actions.push("system")
      break
  }
  
  return actions
}
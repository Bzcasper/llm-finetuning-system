# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a monorepo containing reference implementations of Model Context Protocol (MCP) servers. MCP enables Large Language Models to securely access tools and data sources through a standardized protocol.

## Commands

### Root Level Commands
- `npm run build` - Build all workspaces (runs build in each server directory)
- `npm run watch` - Watch mode for all workspaces
- `npm run publish-all` - Publish all packages to npm (requires access)
- `npm run link-all` - Link all workspaces locally

### Individual Server Commands
Each server in `src/` has its own package.json with specific commands:

**TypeScript Servers (everything, filesystem, memory, sequentialthinking):**
- `npm run build` - Compile TypeScript to JavaScript
- `npm run watch` - Watch mode for development
- `npm run prepare` - Build before publishing
- `npm run start` - Run the MCP server (stdio transport)

**Python Servers (fetch, git, time):**
- Testing: `pytest` or `pytest tests/` (requires pytest to be installed)
- Build: Uses `uv` package manager and `hatchling` build system
- Run: `python -m mcp_server_[name]` or use the installed console script

**Testing:**
- Filesystem server: `npm test` (uses Jest)
- Python servers: `pytest tests/`

## Architecture

### Core Structure
- **Monorepo with workspaces**: Each server is in `src/[server-name]/` with independent package.json
- **Mixed languages**: TypeScript servers use npm, Python servers use uv/pip
- **MCP SDK**: All servers use official MCP SDKs (@modelcontextprotocol/sdk for TS, mcp for Python)

### Server Types
- **everything**: Reference server demonstrating all MCP features (tools, resources, prompts)
- **filesystem**: Secure file operations with configurable directory access
- **memory**: Knowledge graph-based persistent memory system
- **git**: Git repository operations (Python)
- **fetch**: Web content fetching and conversion (Python)
- **time**: Time and timezone utilities (Python)
- **sequentialthinking**: Dynamic problem-solving through thought sequences

### Transport Layers
Servers support multiple transport mechanisms:
- **stdio**: Standard input/output (default)
- **SSE**: Server-Sent Events
- **HTTP**: Direct HTTP endpoints

### Key Architectural Patterns

**TypeScript Servers:**
- Use Zod for input validation and schema generation
- Follow ES modules with `.js` extensions in imports
- Server class from MCP SDK with transport layers
- Tool registration with JSON schema validation
- Error handling with try/catch and clear error messages

**Python Servers:**
- Use Pydantic for data validation
- Click for CLI argument parsing
- GitPython for git operations (git server)
- Async/await patterns for server operations

### Configuration
- **Allowed directories**: Filesystem server requires directory arguments for security
- **Transport selection**: Everything server supports multiple transport modes via CLI args
- **Environment-based**: Many servers can be configured via environment variables

### Development Workflow
1. Navigate to specific server directory: `cd src/[server-name]`
2. Install dependencies: `npm install` (TS) or `uv sync` (Python)
3. Build: `npm run build` (TS) or automatic with uv (Python)
4. Run: `npm start` (TS) or `python -m mcp_server_[name]` (Python)
5. Test: `npm test` (if available) or `pytest tests/`

### Security Considerations
- Filesystem server enforces directory boundaries
- All servers validate inputs using schemas
- Transport security handled by MCP protocol layer
- Git server provides read-only operations by default
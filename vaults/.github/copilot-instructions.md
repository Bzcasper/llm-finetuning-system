<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# HashiCorp Vault Project Instructions

This project sets up HashiCorp Vault for secrets management using Docker Compose.

## Key Components

- **Docker Compose**: Uses `hashicorp/vault:latest` image
- **Configuration**: Vault config in `config/vault.hcl`
- **Policies**: HCL policy files in `config/policies/`
- **Scripts**: Initialization script at `scripts/init-vault.sh`

## Development Guidelines

- All Docker commands should use `sudo` for permission handling
- Use `docker exec vault vault ...` for CLI operations
- Root token for dev: `root-token`
- Vault address: `http://127.0.0.1:8200`

## Common Patterns

- KV v2 secrets engine at path `secret/`
- Policies follow principle of least privilege
- Use VS Code tasks for container management

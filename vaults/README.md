# HashiCorp Vault Setup

A complete HashiCorp Vault setup using Docker Compose for secrets management.

## Project Structure

```
vaults/
├── docker-compose.yml      # Docker Compose configuration
├── config/
│   ├── vault.hcl          # Vault server configuration
│   └── policies/          # Vault policies
│       ├── read-secrets.hcl
│       └── write-secrets.hcl
├── scripts/
│   └── init-vault.sh      # Vault initialization script
└── .vscode/
    └── tasks.json         # VS Code tasks
```

## Quick Start

### 1. Start Vault Server

Use VS Code Task:

- Press `Ctrl+Shift+P` → Run Task → "Start Vault Server"

Or via command line:

```bash
sudo docker-compose up -d
```

### 2. Initialize Vault

Run the initialization script:

```bash
sudo ./scripts/init-vault.sh
```

This script will:

- Enable KV v2 secrets engine
- Apply read/write policies
- Create an example secret

### 3. Access Vault

- **Web UI**: http://localhost:8200
- **Root Token**: `root-token`

## Manual CLI Usage

### Set Environment Variables

```bash
export VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_TOKEN="root-token"
```

### Basic Operations

**Store a secret:**

```bash
sudo docker exec vault vault kv put secret/data/myapp/db username="dbuser" password="secret123"
```

**Read a secret:**

```bash
sudo docker exec vault vault kv get secret/data/myapp/db
```

**List secrets:**

```bash
sudo docker exec vault vault kv list secret/data
```

## Policies

### Read-only Policy (`read-secrets.hcl`)

Allows reading secrets under `secret/data/*`

### Write Policy (`write-secrets.hcl`)

Allows creating/updating secrets under `secret/data/*`

## VS Code Tasks

- **Start Vault Server**: `docker-compose up -d`
- **Stop Vault Server**: `docker-compose down`

## Security Notes

⚠️ **This setup is for development only!**

- Uses hardcoded root token
- TLS disabled
- Not suitable for production

## Troubleshooting

### Permission Denied Errors

If you see Docker permission errors, add your user to the docker group:

```bash
sudo usermod -aG docker $USER
```

Then logout and login again.

### Container Not Found

Ensure the Vault container is running:

```bash
sudo docker ps
```

### Access Vault Logs

```bash
sudo docker logs vault
```

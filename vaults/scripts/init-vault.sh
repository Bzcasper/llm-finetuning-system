#!/usr/bin/env bash

# Check if user is in docker group
if ! groups $USER | grep -q '\bdocker\b'; then
    echo "Adding user to docker group..."
    echo "8040" | sudo -S usermod -aG docker $USER
    echo "Please logout and login again for group changes to take effect."
    echo "For now, we'll continue with sudo..."
    USE_SUDO="sudo"
else
    USE_SUDO=""
fi

# Stop any existing vault container
echo "Stopping any existing Vault containers..."
$USE_SUDO docker-compose down

# Start Vault container
echo "Starting Vault container..."
$USE_SUDO docker-compose up -d

echo "Waiting for Vault to be ready..."
sleep 10

# Wait for Vault to be responsive
echo "Checking if Vault is ready..."
for i in {1..30}; do
    if $USE_SUDO docker exec -e VAULT_ADDR='http://127.0.0.1:8200' vault vault status >/dev/null 2>&1; then
        echo "Vault is ready!"
        break
    fi
    echo "Waiting for Vault... ($i/30)"
    sleep 2
done

echo "Initializing Vault..."

# Set Vault connection variables
export VAULT_ADDR="http://127.0.0.1:8200"
export VAULT_TOKEN="root-token"

# Enable KV v2 secrets engine inside container
echo "Enabling KV v2 secrets engine..."
$USE_SUDO docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault vault secrets enable -path=secret kv-v2

# Apply policies inside container  
echo "Applying policies..."
$USE_SUDO docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault vault policy write read-secrets /vault/config/policies/read-secrets.hcl
$USE_SUDO docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault vault policy write write-secrets /vault/config/policies/write-secrets.hcl

# Create example secret inside container
echo "Creating example secret..."
$USE_SUDO docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault vault kv put secret/myapp/config username="admin" password="p@ssw0rd"

echo ""
echo "✅ Vault initialized successfully!"
echo "🌐 Access Vault at: http://localhost:8200"
echo "🔑 Root token: root-token"
echo ""
echo "Test commands:"
echo "  $USE_SUDO docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault vault kv get secret/myapp/config"
echo "  $USE_SUDO docker exec -e VAULT_ADDR='http://127.0.0.1:8200' -e VAULT_TOKEN='root-token' vault vault kv list secret/"

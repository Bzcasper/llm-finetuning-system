#!/usr/bin/env bash

# Helper script to run Docker commands with automatic sudo password
# Password: 8040

DOCKER_PASS="8040"

run_docker() {
    echo "$DOCKER_PASS" | sudo -S "$@"
}

case "$1" in
    "start")
        echo "Starting Vault server..."
        run_docker docker-compose up -d
        ;;
    "stop")
        echo "Stopping Vault server..."
        run_docker docker-compose down
        ;;
    "ps"|"containers")
        echo "Vault containers:"
        run_docker docker ps -a --filter name=vault
        ;;
    "logs")
        echo "Vault logs:"
        run_docker docker logs vault -f
        ;;
    "init")
        echo "Initializing Vault..."
        ./scripts/init-vault.sh
        ;;
    "status")
        echo "Docker daemon status:"
        run_docker systemctl status docker --no-pager
        ;;
    *)
        echo "Vault Docker Helper"
        echo "Usage: $0 {start|stop|ps|logs|init|status}"
        echo ""
        echo "Commands:"
        echo "  start      - Start Vault server"
        echo "  stop       - Stop Vault server" 
        echo "  ps         - Show Vault containers"
        echo "  logs       - Follow Vault logs"
        echo "  init       - Initialize Vault"
        echo "  status     - Check Docker daemon"
        ;;
esac

#!/usr/bin/env bash
# ==============================================================================
# ORX AGENTS — 1-CLICK ORACLE CLOUD (PHOENIX / ARIZONA) DEPLOYMENT SCRIPT
# Domain: https://agents.orxlabs.com
# ==============================================================================

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RESET='\033[0m'

echo -e "${BOLD}${CYAN}==============================================================${RESET}"
echo -e "${BOLD}${CYAN}   🚀 Deploying ORX Agents to Oracle Cloud Arizona (Ampere A1)${RESET}"
echo -e "${BOLD}${CYAN}   Target: https://agents.orxlabs.com${RESET}"
echo -e "${BOLD}${CYAN}==============================================================${RESET}\n"

# 1. Update OS packages & install Docker if not present
echo -e "${YELLOW}Step 1/5: Checking Docker and system prerequisites...${RESET}"
if ! command -v docker &> /dev/null; then
    echo "Installing Docker engine and docker compose plugin..."
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl gnupg lsb-release
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo -e "${GREEN}✓ Docker installed successfully.${RESET}"
else
    echo -e "${GREEN}✓ Docker is already installed.${RESET}"
fi

# 2. Prepare persistent data directories
echo -e "\n${YELLOW}Step 2/5: Provisioning persistent tenant storage...${RESET}"
mkdir -p data/projects data/recordings data/transcripts

# 3. Environment check
echo -e "\n${YELLOW}Step 3/5: Checking environment variables...${RESET}"
if [ ! -f .env ]; then
    if [ -f deploy/production.env ]; then
        cp deploy/production.env .env
        echo -e "${GREEN}✓ Initialized .env from deploy/production.env${RESET}"
    else
        echo -e "${YELLOW}⚠️  No .env file found. Please create one with your API keys.${RESET}"
    fi
fi

# 4. Build and run stack with Docker Compose
echo -e "\n${YELLOW}Step 4/5: Building and starting containerized stack...${RESET}"
docker compose down --remove-orphans || true
docker compose up -d --build

# 5. Wait for healthcheck
echo -e "\n${YELLOW}Step 5/5: Verifying container health...${RESET}"
sleep 8
for i in {1..12}; do
    if curl -s -f http://localhost:7860/health > /dev/null; then
        echo -e "${GREEN}${BOLD}✓ ORX Voice Agent is healthy and operational on port 7860!${RESET}"
        break
    else
        echo "Waiting for service to become healthy ($i/12)..."
        sleep 4
    fi
done

echo -e "\n${BOLD}${GREEN}==============================================================${RESET}"
echo -e "${BOLD}${GREEN}   ✅ Deployment Complete!${RESET}"
echo -e "${BOLD}${GREEN}   Admin Dashboard: https://agents.orxlabs.com/dashboard${RESET}"
echo -e "${BOLD}${GREEN}   Client Portal:   https://agents.orxlabs.com/portal${RESET}"
echo -e "${BOLD}${GREEN}   Onboarding Flow: https://agents.orxlabs.com/subscribe${RESET}"
echo -e "${BOLD}${GREEN}==============================================================${RESET}"

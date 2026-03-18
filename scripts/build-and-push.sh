#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Build and push Docker images to Docker Hub

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Docker Hub username
DOCKER_USERNAME="${DOCKER_USERNAME:-dongfangzan}"
VERSION="${VERSION:-latest}"

echo -e "${GREEN}Building and pushing Docker images...${NC}"
echo "Docker Hub Username: $DOCKER_USERNAME"
echo "Version: $VERSION"
echo ""

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if user is logged in to Docker Hub
if ! docker info | grep -q "Username"; then
    echo -e "${YELLOW}Not logged in to Docker Hub. Please login:${NC}"
    docker login
fi

echo -e "${GREEN}1. Building OA2A proxy image...${NC}"
docker build -t "$DOCKER_USERNAME/local-openai2anthropic:$VERSION" -t "$DOCKER_USERNAME/local-openai2anthropic:latest" .

echo ""
echo -e "${GREEN}2. Building Claude Code image...${NC}"
docker build -t "$DOCKER_USERNAME/claude-code:$VERSION" -t "$DOCKER_USERNAME/claude-code:latest" ./claude-code

echo ""
echo -e "${GREEN}3. Pushing OA2A proxy image...${NC}"
docker push "$DOCKER_USERNAME/local-openai2anthropic:$VERSION"
docker push "$DOCKER_USERNAME/local-openai2anthropic:latest"

echo ""
echo -e "${GREEN}4. Pushing Claude Code image...${NC}"
docker push "$DOCKER_USERNAME/claude-code:$VERSION"
docker push "$DOCKER_USERNAME/claude-code:latest"

echo ""
echo -e "${GREEN}✅ All images built and pushed successfully!${NC}"
echo ""
echo "Images:"
echo "  - $DOCKER_USERNAME/local-openai2anthropic:$VERSION"
echo "  - $DOCKER_USERNAME/claude-code:$VERSION"

#!/bin/bash

# Deployment script for Attendance Monitoring System
# Usage: ./deploy.sh [env]
# where [env] can be 'dev' or 'prod' (default: dev)

set -e

# Default environment
ENV=${1:-dev}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Starting deployment for ${ENV} environment...${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating from .env.example...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}ℹ️  Please update the .env file with your configuration and run the script again.${NC}"
    exit 1
fi

# Load environment variables
echo -e "${GREEN}✓ Loading environment variables...${NC}"
source .env

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Docker and Docker Compose
if ! command_exists docker || ! command_exists docker-compose; then
    echo -e "${YELLOW}⚠️  Docker and/or Docker Compose is not installed. Please install them first.${NC}"
    exit 1
fi

# Build and start services
echo -e "${GREEN}🚀 Building and starting services...${NC}"

if [ "$ENV" = "prod" ]; then
    # Production deployment
    echo -e "${GREEN}🚀 Starting production deployment...${NC}"    
    # Build and start production containers
    docker-compose -f docker-compose.yml up -d --build
    
    # Run database migrations (if needed)
    # echo -e "${GREEN}🚀 Running database migrations...${NC}"
    # docker-compose exec backend alembic upgrade head
    
    echo -e "${GREEN}✅ Production deployment completed successfully!${NC}"
    echo -e "${GREEN}🌐 Application is running at http://localhost:5000${NC}"
else
    # Development deployment
    echo -e "${GREEN}🚀 Starting development environment...${NC}"
    
    # Start development containers
    docker-compose -f docker-compose.yml up --build
    
    echo -e "${GREEN}✅ Development environment is ready!${NC}"
    echo -e "${GREEN}🌐 Frontend is running at http://localhost:3000${NC}"
    echo -e "${GREEN}🚀 Backend API is running at http://localhost:5000${NC}"
fi

echo -e "${GREEN}✨ Deployment completed successfully!${NC}"

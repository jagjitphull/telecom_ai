#!/bin/bash

# Telecom Log Analysis Setup Script
# This script helps set up the application and Ollama

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
cat << "EOF"
╔════════════════════════════════════════╗
║  Telecom Log Analysis - Setup Script   ║
╚════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Check Python
echo -e "${BLUE}[1/5] Checking Python...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓ $PYTHON_VERSION found${NC}"
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Install dependencies
echo -e "${BLUE}[2/5] Installing Python dependencies...${NC}"
python3 -m pip install -q -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check Ollama
echo -e "${BLUE}[3/5] Checking Ollama...${NC}"
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓ Ollama found${NC}"
    
    # Check if Ollama is running
    echo -e "${YELLOW}Note: Make sure Ollama is running in another terminal${NC}"
    echo -e "${YELLOW}      Run: ollama serve${NC}"
else
    echo -e "${YELLOW}⚠ Ollama not found${NC}"
    echo "Please install Ollama from https://ollama.ai/"
    echo ""
    echo "After installation, run these commands in separate terminals:"
    echo "  Terminal 1: ollama serve"
    echo "  Terminal 2: python app.py"
fi

# Download models
echo -e "${BLUE}[4/5] Checking Ollama models...${NC}"
read -p "Do you want to pull recommended models? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Note: This requires Ollama to be running!${NC}"
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Pulling mistral (fast)..."
        ollama pull mistral || echo -e "${YELLOW}⚠ Could not pull mistral (is Ollama running?)${NC}"
        
        read -p "Pull llama2 as well? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Pulling llama2 (balanced)..."
            ollama pull llama2 || echo -e "${YELLOW}⚠ Could not pull llama2${NC}"
        fi
    fi
fi

# Create directories
echo -e "${BLUE}[5/5] Creating directories...${NC}"
mkdir -p uploads reports
echo -e "${GREEN}✓ Directories created${NC}"

# Success message
echo ""
echo -e "${GREEN}"
cat << "EOF"
╔════════════════════════════════════════╗
║  Setup Complete! ✓                     ║
╚════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "${BLUE}Quick Start:${NC}"
echo "1. In Terminal 1, start Ollama:"
echo -e "   ${YELLOW}ollama serve${NC}"
echo ""
echo "2. In Terminal 2, start the app:"
echo -e "   ${YELLOW}python app.py${NC}"
echo ""
echo "3. Open your browser:"
echo -e "   ${YELLOW}http://localhost:5000${NC}"
echo ""
echo -e "${BLUE}Documentation:${NC}"
echo "  Read README.md for detailed instructions"
echo ""

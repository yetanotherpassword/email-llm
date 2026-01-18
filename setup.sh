#!/bin/bash
# Setup script for email-llm

set -e

echo "================================"
echo "Email-LLM Setup"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.10" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.10 or higher is required"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package
echo ""
echo "Installing email-llm..."
pip install -e .

# Check for system dependencies
echo ""
echo "Checking system dependencies..."

# Check for Tesseract
if command -v tesseract &> /dev/null; then
    echo "  [OK] Tesseract OCR is installed"
else
    echo "  [WARN] Tesseract OCR not found"
    echo "         Install with: sudo apt install tesseract-ocr"
fi

# Copy example env if .env doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env from example..."
    cp .env.example .env
    echo "  Please edit .env to configure your settings"
fi

# Create data directories
echo ""
echo "Creating data directories..."
mkdir -p data/chroma data/attachments data/yolo_cache

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Edit .env to set your Thunderbird profile path"
echo ""
echo "3. Start LM Studio or Ollama:"
echo "   - LM Studio: Start the app and load a model"
echo "   - Ollama: ollama serve && ollama pull mistral"
echo ""
echo "4. Check your configuration:"
echo "   email-llm check"
echo ""
echo "5. Index your emails:"
echo "   email-llm index"
echo ""
echo "6. Search your emails:"
echo "   email-llm search 'find invoices from last month'"
echo ""
echo "7. Or start the web UI:"
echo "   email-llm serve"
echo ""

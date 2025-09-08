#!/bin/bash
# Project Nova - Quick Setup Script for Unix/Linux/macOS

echo "============================================================"
echo "PROJECT NOVA - ENVIRONMENT SETUP (Unix/Linux/macOS)"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC} $1"
}

# Check if Python 3 is installed
echo "🔍 Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_status "Found: $PYTHON_VERSION"
elif command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    if [[ $PYTHON_VERSION == *"Python 3"* ]]; then
        PYTHON_CMD="python"
        print_status "Found: $PYTHON_VERSION"
    else
        print_error "Python 3 is required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version (require 3.8+)
PYTHON_VERSION_NUM=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if (( $(echo "$PYTHON_VERSION_NUM < 3.8" | bc -l) )); then
    print_error "Python 3.8+ required. Found: Python $PYTHON_VERSION_NUM"
    exit 1
fi

# Create virtual environment
VENV_NAME="nova_env"
echo ""
echo "🏗️  Creating virtual environment '$VENV_NAME'..."

if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment '$VENV_NAME' already exists."
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf "$VENV_NAME"
    else
        print_info "Using existing virtual environment."
    fi
fi

if [ ! -d "$VENV_NAME" ]; then
    $PYTHON_CMD -m venv "$VENV_NAME"
    if [ $? -eq 0 ]; then
        print_status "Virtual environment created successfully"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source "$VENV_NAME/bin/activate"

if [ $? -eq 0 ]; then
    print_status "Virtual environment activated"
else
    print_error "Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo ""
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "📚 Installing requirements..."

# Check which requirements files are available
if [ -f "requirements-minimal.txt" ] && [ -f "requirements.txt" ]; then
    echo "Available installation options:"
    echo "1. Minimal (faster, core packages only)"
    echo "2. Full (all packages, may take longer)"
    read -p "Choose installation type (1 or 2): " -n 1 -r
    echo
    
    if [[ $REPLY == "1" ]]; then
        REQ_FILE="requirements-minimal.txt"
    else
        REQ_FILE="requirements.txt"
    fi
elif [ -f "requirements-minimal.txt" ]; then
    REQ_FILE="requirements-minimal.txt"
elif [ -f "requirements.txt" ]; then
    REQ_FILE="requirements.txt"
else
    print_error "No requirements files found"
    exit 1
fi

echo "Installing from $REQ_FILE..."
pip install -r "$REQ_FILE"

if [ $? -eq 0 ]; then
    print_status "Requirements installed successfully"
    
    # If minimal was installed, offer to install additional packages
    if [ "$REQ_FILE" == "requirements-minimal.txt" ] && [ -f "requirements.txt" ]; then
        echo ""
        read -p "Install additional packages for full functionality? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Installing additional packages..."
            pip install catboost tensorflow plotly shap fairlearn
        fi
    fi
else
    print_error "Failed to install some requirements"
    
    # Try minimal as fallback
    if [ "$REQ_FILE" == "requirements.txt" ] && [ -f "requirements-minimal.txt" ]; then
        print_info "Trying minimal installation as fallback..."
        pip install -r requirements-minimal.txt
        if [ $? -eq 0 ]; then
            print_status "Minimal requirements installed successfully"
        else
            print_error "Even minimal installation failed"
            exit 1
        fi
    else
        exit 1
    fi
fi

# Verify key packages
echo ""
echo "🔍 Verifying installation..."
KEY_PACKAGES=("pandas" "numpy" "scikit-learn" "xgboost" "lightgbm" "tensorflow")

for package in "${KEY_PACKAGES[@]}"; do
    python -c "import $package" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_status "$package"
    else
        print_warning "$package (failed to import)"
    fi
done

# Create activation script
echo ""
echo "📝 Creating activation script..."
cat > activate_nova.sh << 'EOF'
#!/bin/bash
echo "Activating Project Nova environment..."
source nova_env/bin/activate

echo ""
echo "✅ Project Nova environment activated!"
echo ""
echo "Available commands:"
echo "  python quick_start.py          - Quick pipeline test"
echo "  python main_training_pipeline.py - Full pipeline with tuning"
echo "  python deploy_model.py         - Deploy trained models"
echo "  python project_summary.py      - Show project overview"
echo ""

# Keep shell open
exec "$SHELL"
EOF

chmod +x activate_nova.sh
print_status "Created activate_nova.sh"

# Display completion message
echo ""
echo "============================================================"
echo "🎉 PROJECT NOVA ENVIRONMENT SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Activate the environment:"
echo "   source nova_env/bin/activate"
echo "   OR run: ./activate_nova.sh"
echo ""
echo "2. Test the setup:"
echo "   python quick_start.py"
echo ""
echo "3. Run full pipeline:"
echo "   python main_training_pipeline.py"
echo ""
echo "💡 TIP: Always activate the virtual environment before running scripts!"
echo "============================================================"

# Ask if user wants to run quick start
echo ""
read -p "Would you like to run the quick start now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "🚀 Running quick start..."
    python quick_start.py
fi
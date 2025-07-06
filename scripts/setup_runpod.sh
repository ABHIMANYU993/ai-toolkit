#!/bin/bash

# RunPod FLUX.1-dev LoRA Training Setup Script
# This script automates the setup process for RunPod environments

echo "🚀 Starting RunPod FLUX.1-dev LoRA Training Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. This is typical for RunPod containers."
fi

# Step 1: System updates
print_status "Step 1: Updating system packages..."
apt update && apt upgrade -y

# Step 2: Install required system packages
print_status "Step 2: Installing system dependencies..."
apt install -y git wget curl nano vim htop

# Step 3: Check for existing ai-toolkit
if [ -d "/workspace/ai-toolkit" ]; then
    print_warning "ai-toolkit directory already exists. Skipping clone."
    cd /workspace/ai-toolkit
else
    print_status "Step 3: Cloning AI Toolkit repository..."
    cd /workspace
    git clone https://github.com/ostris/ai-toolkit.git
    cd ai-toolkit
fi

# Step 4: Initialize submodules
print_status "Step 4: Initializing git submodules..."
git submodule update --init --recursive

# Step 5: Create virtual environment
print_status "Step 5: Creating Python virtual environment..."
python -m venv venv

# Step 6: Activate virtual environment
print_status "Step 6: Activating virtual environment..."
source venv/bin/activate

# Step 7: Install PyTorch
print_status "Step 7: Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 8: Install requirements
print_status "Step 8: Installing Python requirements..."
pip install -r requirements.txt

# Step 9: Optional packages
print_status "Step 9: Installing optional packages..."
pip install --upgrade accelerate transformers diffusers huggingface_hub

# Step 10: Create dataset directory
print_status "Step 10: Creating dataset directory..."
mkdir -p /workspace/ai-toolkit/dataset

# Step 11: Test CUDA availability
print_status "Step 11: Testing CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || print_error "CUDA test failed"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')" || print_error "CUDA device count failed"

# Step 12: Check GPU memory
print_status "Step 12: Checking GPU memory..."
nvidia-smi || print_error "nvidia-smi failed"

# Step 13: Create sample config
print_status "Step 13: Creating sample training config..."
if [ ! -f "config/my_flux_training.yaml" ]; then
    cp config/examples/train_lora_flux_runpod.yaml config/my_flux_training.yaml
    print_status "Sample config created at config/my_flux_training.yaml"
else
    print_warning "config/my_flux_training.yaml already exists"
fi

# Step 14: Display next steps
print_status "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Upload your dataset to: /workspace/ai-toolkit/dataset/"
echo "2. Create caption .txt files for each image"
echo "3. Login to Hugging Face: huggingface-cli login"
echo "4. Edit config: nano config/my_flux_training.yaml"
echo "5. Start training: python run.py config/my_flux_training.yaml"
echo ""
echo "📚 For detailed instructions, see: docs/RUNPOD_FLUX_TRAINING_GUIDE.md"
echo ""
echo "🎯 Quick start command:"
echo "   python flux_train_ui.py"
echo ""

# Step 15: Display environment info
print_status "Environment Information:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Not available')"
echo "Working directory: $(pwd)"
echo "Virtual environment: $VIRTUAL_ENV"

print_status "Setup script completed successfully! 🎉"
#!/bin/bash

# RunPod FLUX.1-dev LoRA Training Demo Script
# This script demonstrates the complete setup and provides examples

echo "🎯 RunPod FLUX.1-dev LoRA Training Complete Solution"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "flux_train_ui.py" ]; then
    echo "❌ Please run this script from the ai-toolkit directory"
    exit 1
fi

echo "📋 Available Resources:"
echo ""

echo "1. 📖 Complete Training Guide:"
echo "   docs/RUNPOD_FLUX_TRAINING_GUIDE.md"
echo "   - Step-by-step instructions"
echo "   - Troubleshooting section"
echo "   - Best practices"
echo ""

echo "2. ⚡ Quick Reference:"
echo "   docs/RUNPOD_QUICK_REFERENCE.md"
echo "   - Commands cheat sheet"
echo "   - Common issues & solutions"
echo ""

echo "3. 🛠️ Setup Automation:"
echo "   scripts/setup_runpod.sh"
echo "   - One-command environment setup"
echo "   - Automatic dependency installation"
echo ""

echo "4. ⚙️ Pre-configured Training:"
echo "   config/examples/train_lora_flux_runpod.yaml"
echo "   - Optimized for RunPod"
echo "   - Detailed comments"
echo ""

echo "5. 📓 Interactive Notebook:"
echo "   notebooks/RunPod_FLUX_Training.ipynb"
echo "   - Step-by-step training"
echo "   - Visual progress monitoring"
echo ""

echo "🚀 Quick Start Options:"
echo ""

echo "Option 1 - Automated Setup:"
echo "  bash scripts/setup_runpod.sh"
echo ""

echo "Option 2 - Manual Setup:"
echo "  1. huggingface-cli login"
echo "  2. mkdir -p dataset"
echo "  3. # Upload images and captions"
echo "  4. cp config/examples/train_lora_flux_runpod.yaml config/my_training.yaml"
echo "  5. python run.py config/my_training.yaml"
echo ""

echo "Option 3 - Gradio UI:"
echo "  python flux_train_ui.py"
echo ""

echo "Option 4 - Jupyter Notebook:"
echo "  jupyter notebook notebooks/RunPod_FLUX_Training.ipynb"
echo ""

echo "📊 Training Information:"
echo "- Minimum GPU: 24GB VRAM"
echo "- Recommended: A40 (48GB) or A100 (40GB)"
echo "- Training time: 1-4 hours"
echo "- Cost: ~$0.50-2.00 per session"
echo "- Dataset: 4-30 images recommended"
echo ""

echo "📁 Expected File Structure:"
echo "ai-toolkit/"
echo "├── dataset/"
echo "│   ├── image1.jpg"
echo "│   ├── image1.txt"
echo "│   └── ..."
echo "├── config/"
echo "│   └── my_training.yaml"
echo "└── output/"
echo "    └── my_model/"
echo "        ├── my_model.safetensors"
echo "        └── samples/"
echo ""

echo "🔧 Common RunPod Commands:"
echo "# Check GPU:"
echo "nvidia-smi"
echo ""
echo "# Monitor training:"
echo "tail -f output/*/logs/training.log"
echo ""
echo "# List outputs:"
echo "ls -la output/"
echo ""

echo "📚 Additional Resources:"
echo "- AI Toolkit: https://github.com/ostris/ai-toolkit"
echo "- FLUX.1-dev: https://huggingface.co/black-forest-labs/FLUX.1-dev"
echo "- RunPod: https://docs.runpod.io/"
echo ""

echo "✅ Everything is ready! Choose your preferred method above."
echo "💡 Pro tip: Start with the automated setup script for fastest results."
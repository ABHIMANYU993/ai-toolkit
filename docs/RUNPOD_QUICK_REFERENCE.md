# RunPod FLUX.1-dev LoRA Training Quick Reference

## 🚀 One-Command Setup
```bash
wget -qO- https://raw.githubusercontent.com/ostris/ai-toolkit/main/scripts/setup_runpod.sh | bash
```

## 📋 RunPod Configuration
- **Template:** `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
- **GPU:** 24GB+ VRAM (A40, A100, RTX 4090)
- **Storage:** 120GB+ recommended
- **Memory:** 32GB+ system RAM

## ⚡ Quick Start Commands
```bash
# 1. Clone and setup
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit && bash scripts/setup_runpod.sh

# 2. Login to Hugging Face
huggingface-cli login

# 3. Prepare dataset
mkdir -p dataset
# Upload images and create .txt caption files

# 4. Start training
cp config/examples/train_lora_flux_runpod.yaml config/my_training.yaml
python run.py config/my_training.yaml
```

## 🎯 Alternative Methods

### Gradio UI
```bash
python flux_train_ui.py
# Access at http://localhost:7860
```

### Jupyter Notebook
```bash
jupyter notebook notebooks/RunPod_FLUX_Training.ipynb
```

## 📊 Training Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| Steps | 1000 | Total training steps |
| Learning Rate | 1e-4 | How fast the model learns |
| LoRA Rank | 16 | Model complexity (4-128) |
| Batch Size | 1 | For 24GB VRAM |
| Images | 4-30 | Dataset size |

## 🛠️ Common Issues & Solutions

### CUDA Out of Memory
```yaml
model:
  low_vram: true
  quantize: true
train:
  gradient_checkpointing: true
  batch_size: 1
```

### Permission Errors
```bash
chmod -R 755 /workspace/ai-toolkit/
```

### Model Download Issues
```bash
rm -rf ~/.cache/huggingface/
huggingface-cli login
```

## 📁 File Structure
```
ai-toolkit/
├── config/
│   ├── examples/
│   │   └── train_lora_flux_runpod.yaml
│   └── my_training.yaml
├── dataset/
│   ├── image1.jpg
│   ├── image1.txt
│   └── ...
├── output/
│   └── my_model/
│       ├── my_model.safetensors
│       └── samples/
└── docs/
    └── RUNPOD_FLUX_TRAINING_GUIDE.md
```

## 🔍 Monitor Training
```bash
# Check GPU usage
nvidia-smi

# View training logs
tail -f output/my_model/logs/training.log

# List generated samples
ls -la output/my_model/samples/
```

## 📤 Download Results
Your trained model will be saved as:
- `output/my_model/my_model.safetensors` (final model)
- `output/my_model/samples/` (test images)

## 🎨 Using Your Model
Compatible with:
- ComfyUI
- Automatic1111
- Diffusers library
- Any FLUX-compatible interface

## 💡 Tips
- Start with 1000 steps, adjust based on results
- Use high-quality, diverse images
- Monitor samples during training
- Save checkpoints frequently
- Consider trigger words for specific activation

## 📚 Resources
- [Detailed Guide](docs/RUNPOD_FLUX_TRAINING_GUIDE.md)
- [AI Toolkit Repo](https://github.com/ostris/ai-toolkit)
- [FLUX.1-dev Model](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [RunPod Docs](https://docs.runpod.io/)

---
*Training typically takes 1-4 hours. Cost: ~$0.50-2.00 per training session*
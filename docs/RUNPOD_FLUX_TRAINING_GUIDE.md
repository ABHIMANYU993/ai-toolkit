# Complete RunPod FLUX1.dev LoRA Training Guide

This comprehensive guide will walk you through the entire process of training a FLUX1.dev LoRA model on RunPod from start to finish.

## Overview

FLUX1.dev is a state-of-the-art text-to-image model that can be fine-tuned using LoRA (Low-Rank Adaptation) to learn specific styles, characters, or concepts. This guide focuses on using RunPod's cloud GPU infrastructure for training.

## Prerequisites

- RunPod account with GPU access
- Hugging Face account with READ token
- Dataset of images (4-30 images recommended)
- Basic understanding of command-line interfaces

## Step 1: RunPod Setup

### 1.1 Choose Your RunPod Configuration

**Recommended Template:** `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`

**Minimum Requirements:**
- **GPU:** Minimum 24GB VRAM (A40, A100, RTX 4090, or similar)
- **CPU:** 8+ vCPUs
- **RAM:** 32GB+ system RAM
- **Storage:** 120GB+ (for model weights, datasets, and outputs)

**Recommended Configuration ($0.5-1.0/hr):**
- 1x A40 (48GB VRAM) or 1x A100 (40GB VRAM)
- 16-19 vCPUs
- 64-100GB RAM
- 120GB Disk Space
- 120GB Pod Volume

### 1.2 Launch Your RunPod Instance

1. Log into your RunPod account
2. Click "Create Pod"
3. Select the recommended template
4. Configure your hardware as specified above
5. Set **Start Jupyter Notebook** to enabled
6. Add these environment variables:
   ```
   JUPYTER_PASSWORD=your_secure_password
   ```
7. Launch the pod

## Step 2: Environment Setup

### 2.1 Connect to Your Pod

Once your pod is running, connect via:
- **Jupyter Notebook:** Use the provided URL
- **SSH:** Use the provided SSH command
- **Web Terminal:** Available through RunPod interface

### 2.2 Initial Setup Commands

Open a terminal and run these commands:

```bash
# Update system
apt update && apt upgrade -y

# Install git if not available
apt install -y git

# Clone the AI Toolkit repository
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit

# Initialize submodules
git submodule update --init --recursive

# Create Python virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch (CUDA 12.1 compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install -r requirements.txt

# Optional: Install additional packages if needed
pip install --upgrade accelerate transformers diffusers huggingface_hub
```

### 2.3 Verify Installation

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

# Check GPU memory
nvidia-smi
```

## Step 3: Hugging Face Authentication

### 3.1 Get Hugging Face Token

1. Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with **READ** permissions
3. Accept the FLUX.1-dev license at [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)

### 3.2 Login to Hugging Face

```bash
# Login using the CLI
huggingface-cli login

# When prompted, paste your READ token
# Choose 'n' for not adding token as git credential
```

### 3.3 Create Environment File

```bash
# Create .env file with your token
echo "HF_TOKEN=your_read_token_here" > .env
```

## Step 4: Dataset Preparation

### 4.1 Create Dataset Directory

```bash
# Create dataset folder
mkdir -p /workspace/ai-toolkit/dataset
cd /workspace/ai-toolkit/dataset
```

### 4.2 Upload Your Images

**Option A: Using Jupyter Notebook**
1. Open Jupyter Notebook
2. Navigate to the dataset folder
3. Upload your images (.jpg, .jpeg, .png)

**Option B: Using wget/curl (for images from URLs)**
```bash
# Example: Download images from URLs
wget -O image1.jpg "https://example.com/image1.jpg"
wget -O image2.jpg "https://example.com/image2.jpg"
```

**Option C: Using RunPod's file upload feature**
1. Use RunPod's web interface to upload files
2. Extract to the dataset folder

### 4.3 Create Caption Files

For each image, create a corresponding `.txt` file with the same name:

```bash
# Example: if you have image1.jpg, create image1.txt
echo "a person in a red jacket standing outdoors" > image1.txt
echo "a person wearing sunglasses in a park" > image2.txt
```

**Caption Tips:**
- Use descriptive, specific captions
- Include your trigger word if using one
- Keep captions concise but informative
- Use `[trigger]` in captions if you want automatic replacement

### 4.4 Verify Dataset Structure

```bash
# List your dataset files
ls -la /workspace/ai-toolkit/dataset/

# Should show pairs like:
# image1.jpg, image1.txt
# image2.jpg, image2.txt
# etc.
```

## Step 5: Training Configuration

### 5.1 Copy Example Config

```bash
# Copy the example FLUX config
cp config/examples/train_lora_flux_24gb.yaml config/my_flux_training.yaml
```

### 5.2 Edit Configuration

Edit the config file with your preferred editor:

```bash
# Using nano
nano config/my_flux_training.yaml

# Or using vim
vim config/my_flux_training.yaml
```

### 5.3 Key Configuration Settings

Here's a sample configuration with explanations:

```yaml
---
job: extension
config:
  name: "my_flux_lora_v1"  # Change this to your model name
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0
      
      # Optional: Add trigger word
      trigger_word: "your_trigger_word"  # Uncomment and set if using
      
      network:
        type: "lora"
        linear: 16          # LoRA rank (4-128, higher = more capacity)
        linear_alpha: 16    # Usually same as linear
        
      save:
        dtype: float16
        save_every: 250     # Save checkpoint every N steps
        max_step_saves_to_keep: 4
        push_to_hub: false  # Set to true if you want to upload to HF
        
      datasets:
        - folder_path: "/workspace/ai-toolkit/dataset"  # Your dataset path
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [512, 768, 1024]
          
      train:
        batch_size: 1
        steps: 1000         # Total training steps (500-4000 recommended)
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4            # Learning rate
        dtype: bf16
        
        ema_config:
          use_ema: true
          ema_decay: 0.99
          
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true
        low_vram: true      # Enable if using display outputs
        
      sample:
        sampler: "flowmatch"
        sample_every: 250   # Generate samples every N steps
        width: 1024
        height: 1024
        prompts:
          - "a person holding a coffee cup"
          - "a beautiful landscape"
          - "portrait of a person"
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 20
```

## Step 6: Start Training

### 6.1 Run Training Command

```bash
# Make sure you're in the ai-toolkit directory
cd /workspace/ai-toolkit

# Activate virtual environment if not already active
source venv/bin/activate

# Start training
python run.py config/my_flux_training.yaml
```

### 6.2 Monitor Training Progress

The training will output logs showing:
- Current step/total steps
- Loss values
- Memory usage
- ETA (estimated time to completion)
- Sample generations (if enabled)

### 6.3 Training Output

Training outputs are saved in the `output` folder:
- **Model checkpoints:** `output/my_flux_lora_v1/`
- **Sample images:** `output/my_flux_lora_v1/samples/`
- **Logs:** Training logs in the terminal

## Step 7: Using Alternative Training Methods

### 7.1 Using the Gradio UI

For a more user-friendly interface:

```bash
# Start the Gradio UI
python flux_train_ui.py

# Access the UI at: http://localhost:7860
# Or use the RunPod public URL if available
```

### 7.2 Using Jupyter Notebook

You can also use the provided Jupyter notebook:

```bash
# Open the notebook
jupyter notebook notebooks/FLUX_1_dev_LoRA_Training.ipynb
```

## Step 8: Advanced Configuration Options

### 8.1 Performance Tuning

For better performance on RunPod:

```yaml
# In your config file
train:
  gradient_checkpointing: true  # Saves VRAM
  dtype: bf16                   # Faster training
  optimizer: "adamw8bit"        # Memory efficient

model:
  quantize: true               # Reduces VRAM usage
  low_vram: true              # Additional VRAM savings
```

### 8.2 Training Parameters

Adjust these based on your dataset:

- **Steps:** 500-4000 (depends on dataset size)
- **Learning Rate:** 1e-4 to 4e-4
- **LoRA Rank:** 4-128 (higher = more capacity)
- **Batch Size:** Usually 1 for 24GB VRAM

### 8.3 Sampling Configuration

```yaml
sample:
  sample_every: 250           # Frequency of sample generation
  sample_steps: 20           # Quality of samples
  prompts:
    - "your test prompts here"
```

## Step 9: Troubleshooting

### 9.1 Common Issues

**CUDA Out of Memory:**
```bash
# Enable low VRAM mode
low_vram: true

# Reduce batch size
batch_size: 1

# Enable gradient checkpointing
gradient_checkpointing: true
```

**Permission Errors:**
```bash
# Fix permissions
chmod -R 755 /workspace/ai-toolkit/
```

**Model Download Issues:**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
huggingface-cli login
```

### 9.2 Performance Issues

**Slow Training:**
- Ensure you're using the correct GPU
- Check nvidia-smi for GPU utilization
- Verify CUDA installation

**Out of Storage:**
```bash
# Check disk space
df -h

# Clean up if needed
rm -rf output/*/samples/  # Remove old samples
```

## Step 10: Accessing Your Results

### 10.1 Download Trained Model

```bash
# Your trained model will be in:
ls -la output/my_flux_lora_v1/

# Files include:
# - Final model: my_flux_lora_v1.safetensors
# - Intermediate checkpoints: my_flux_lora_v1_000XXX.safetensors
# - Sample images: samples/
```

### 10.2 Using RunPod File Manager

1. Navigate to the `output` folder in RunPod's file manager
2. Download your trained model files
3. Download sample images if needed

### 10.3 Upload to Hugging Face (Optional)

If you set `push_to_hub: true` in your config:

```bash
# Make sure you have a WRITE token
huggingface-cli login

# The model will be automatically uploaded during training
```

## Step 11: Testing Your Model

### 11.1 Local Testing

```bash
# You can test your model using the generated samples
# Check the samples folder for generated images
ls -la output/my_flux_lora_v1/samples/
```

### 11.2 Using the Model

Your trained LoRA can be used with:
- ComfyUI
- Automatic1111
- Diffusers library
- Any FLUX-compatible interface

## Best Practices

1. **Dataset Quality:** Use high-quality, diverse images
2. **Training Steps:** Start with 1000 steps, adjust based on results
3. **Monitor Progress:** Check samples regularly
4. **Save Frequently:** Use reasonable save intervals
5. **Backup:** Download your models regularly

## Cost Optimization

- **Choose appropriate hardware:** Don't over-provision
- **Monitor training:** Stop early if converging
- **Use spot instances:** If available and suitable
- **Batch training:** Train multiple models in sequence

## Conclusion

This guide provides a complete workflow for training FLUX1.dev LoRA models on RunPod. The key is to:

1. Set up your environment correctly
2. Prepare your dataset properly
3. Configure training parameters appropriately
4. Monitor and adjust as needed

Training times typically range from 1-4 hours depending on your dataset size and training steps. With proper setup, you should achieve high-quality results suitable for your specific use case.

## Support and Resources

- **AI Toolkit Repository:** [https://github.com/ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
- **RunPod Documentation:** [https://docs.runpod.io/](https://docs.runpod.io/)
- **FLUX.1-dev Model:** [https://huggingface.co/black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- **Discord Community:** Join the AI Toolkit Discord for support

---

*This guide is maintained by the AI Toolkit community. For updates and improvements, please contribute to the repository.*
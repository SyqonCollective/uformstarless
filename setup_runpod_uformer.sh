#!/bin/bash
# Setup completo UFormer per RunPod A100
# Comando unico per installare tutte le dipendenze

set -e  # Exit on error

echo "🚀 Setting up UFormer Star Removal on RunPod A100"
echo "================================================="

# Update system
echo "📦 Updating system packages..."
apt-get update -qq

# Install system dependencies
echo "🔧 Installing system dependencies..."
apt-get install -y -qq \
    git \
    wget \
    unzip \
    htop \
    nvtop \
    tmux \
    nano \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1

# Install Python packages
echo "🐍 Installing Python packages..."
pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    albumentations \
    opencv-python-headless \
    Pillow \
    numpy \
    scikit-image \
    matplotlib \
    seaborn \
    tqdm \
    tensorboard \
    wandb \
    pyyaml \
    einops \
    timm

# Verify CUDA and PyTorch
echo "✅ Verifying installation..."
python -c "
import torch
import torchvision
import cv2
import albumentations
from skimage import metrics
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print('All packages imported successfully!')
"

# Create workspace structure
echo "📁 Creating workspace structure..."
mkdir -p /workspace/uformer_training/{experiments,logs,data}

echo "🎉 Setup completed successfully!"
echo ""
echo "Ready for UFormer training! Next steps:"
echo "1. Upload your training code and data"
echo "2. Run: python train_uformer.py --config config_uformer.yaml"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

#!/bin/bash
# Setup script per Quick Fix training su RunPod

echo "🚀 Setting up Quick Fix training on RunPod..."

# Check se siamo su RunPod
if [ ! -d "/workspace" ]; then
    echo "❌ This script is for RunPod environment"
    exit 1
fi

cd /workspace/StarLess

# Backup dei file modificati localmente se esistono
if [ -f "quick_fix_uformer.py" ]; then
    cp quick_fix_uformer.py quick_fix_uformer_backup.py
    echo "📁 Backed up existing quick_fix_uformer.py"
fi

echo "📤 Upload these files to RunPod:"
echo "  - train_quick_fix.py"
echo "  - config_quick_fix.yaml" 
echo "  - quick_fix_uformer.py (updated version)"

echo ""
echo "🔍 Checking required files..."

# Check checkpoint
if [ -f "experiments/uformer_debug/checkpoints/best_model.pth" ]; then
    echo "✅ Found checkpoint: best_model.pth"
    CHECKPOINT="experiments/uformer_debug/checkpoints/best_model.pth"
elif [ -f "experiments/uformer_debug/checkpoints/checkpoint_epoch_0004.pth" ]; then
    echo "✅ Found checkpoint: checkpoint_epoch_0004.pth"
    CHECKPOINT="experiments/uformer_debug/checkpoints/checkpoint_epoch_0004.pth"
else
    echo "❌ No checkpoint found in experiments/uformer_debug/checkpoints/"
    echo "Available files:"
    ls -la experiments/uformer_debug/checkpoints/ 2>/dev/null || echo "Directory not found"
    exit 1
fi

# Check training files
for file in "train_quick_fix.py" "config_quick_fix.yaml" "quick_fix_uformer.py"; do
    if [ -f "$file" ]; then
        echo "✅ Found: $file"
    else
        echo "❌ Missing: $file - Upload this file first!"
        exit 1
    fi
done

# Check data directories
if [ -d "train/input" ] && [ -d "train/target" ]; then
    echo "✅ Training data found"
else
    echo "❌ Training data not found (train/input, train/target)"
    exit 1
fi

echo ""
echo "🚀 Ready to start Quick Fix training!"
echo ""
echo "Run this command:"
echo "nohup python train_quick_fix.py --config config_quick_fix.yaml --pretrained $CHECKPOINT > training_quick_fix.log 2>&1 &"
echo ""
echo "Monitor with:"
echo "tail -f training_quick_fix.log"

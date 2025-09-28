#!/bin/bash
# RunPod Setup Script per Enhanced UFormer
# Clona i file direttamente in /workspace invece di creare sottocartella

set -e

echo "🚀 Enhanced UFormer RunPod Setup"
echo "================================="

# Verifica di essere in /workspace
if [[ "$(pwd)" != "/workspace" ]]; then
    echo "⚠️  Cambiando directory in /workspace..."
    cd /workspace
fi

echo "📍 Directory corrente: $(pwd)"

# Pulisci eventuali file esistenti (opzionale - commenta se vuoi mantenere)
echo "🧹 Pulizia workspace (mantiene /workspace/.jupyter se esiste)..."
find /workspace -maxdepth 1 -name ".*" -not -name ".jupyter" -not -name ".." -not -name "." -exec rm -rf {} + 2>/dev/null || true
find /workspace -maxdepth 1 -type f -exec rm -f {} + 2>/dev/null || true
find /workspace -maxdepth 1 -type d -not -name ".jupyter" -exec rm -rf {} + 2>/dev/null || true

# Clone del repository DIRETTAMENTE nella root workspace
echo "📥 Clonazione repository..."
git clone https://github.com/SyqonCollective/uformstarless.git temp_repo

# Sposta tutti i file dalla cartella temp nella root
echo "📦 Spostamento file in /workspace..."
mv temp_repo/* /workspace/
mv temp_repo/.* /workspace/ 2>/dev/null || true  # Sposta anche file nascosti
rm -rf temp_repo

echo "✅ File clonati direttamente in /workspace"

# Verifica che siamo nella directory corretta
echo "📋 Contenuto /workspace:"
ls -la /workspace

# Setup ambiente Python
echo "🐍 Setup ambiente Python..."

# Aggiorna pip
pip install --upgrade pip

# Installa dipendenze principali
echo "📦 Installazione dipendenze..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pillow pyyaml tqdm matplotlib seaborn
pip install opencv-python-headless scikit-image

# Installa dipendenze aggiuntive se requirements.txt esiste
if [[ -f "requirements.txt" ]]; then
    echo "📋 Installazione da requirements.txt..."
    pip install -r requirements.txt
fi

# Verifica installazione PyTorch
echo "🔍 Verifica PyTorch..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Crea directory per esperimenti e checkpoints
echo "📁 Creazione directory di lavoro..."
mkdir -p experiments
mkdir -p checkpoints
mkdir -p data
mkdir -p logs

# Rendi eseguibili gli script shell
echo "🔧 Configurazione permessi..."
chmod +x *.sh 2>/dev/null || true

# Test rapido del modello
echo "🧪 Test rapido del modello..."
python -c "
try:
    from enhanced_uformer import EnhancedUFormerStarRemoval
    print('✅ Enhanced UFormer importato correttamente')
    
    model = EnhancedUFormerStarRemoval(
        embed_dim=64,  # Leggero per test
        window_size=8,
        shifted_window=True
    )
    print('✅ Modello creato correttamente')
    print(f'✅ Parametri totali: {sum(p.numel() for p in model.parameters()):,}')
    
except Exception as e:
    print(f'❌ Errore: {e}')
    print('Verifica che tutti i moduli siano presenti')
"

echo ""
echo "🎉 Setup completato!"
echo "================================="
echo ""
echo "📝 Quick Start Commands:"
echo ""
echo "# Test del modello:"
echo "python demo_enhanced_shifted.py --config config_uformer.yaml --input your_image.jpg"
echo ""
echo "# Training da checkpoint esistente:"
echo "python enhanced_uformer_finetune.py --config config_uformer.yaml --pretrained checkpoint.pth"
echo ""
echo "# Training da zero:"
echo "python enhanced_uformer_finetune.py --config config_uformer.yaml"
echo ""
echo "📊 Configurazione Ottimale:"
echo "- Window Size: 8 (inferenza veloce)"
echo "- Shifted Windows: True (elimina quadretti 8×8)"
echo "- Depths: [2,2,6,2] (contesto esteso)"
echo "- Perceptual Loss: L1 + VGG + SSIM"
echo ""
echo "🚀 Il tuo Enhanced UFormer è pronto!"
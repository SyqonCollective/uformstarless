#!/bin/bash
# Quick Start Script per RunPod - Enhanced UFormer
# Uso: ./runpod_start.sh [train|demo|test]

set -e

MODE=${1:-demo}

echo "🚀 Enhanced UFormer RunPod - Quick Start"
echo "========================================"
echo "Mode: $MODE"
echo ""

# Verifica che siamo nella directory corretta
if [[ ! -f "enhanced_uformer.py" ]]; then
    echo "❌ enhanced_uformer.py non trovato!"
    echo "Assicurati di essere in /workspace e di aver eseguito setup_runpod_enhanced.sh"
    exit 1
fi

case $MODE in
    "demo")
        echo "🎬 Demo Mode - Test con immagine di esempio"
        echo ""
        
        # Crea immagine di test se non esiste
        if [[ ! -f "test_image.jpg" ]]; then
            echo "📸 Creazione immagine di test..."
            python -c "
import torch
import numpy as np
from PIL import Image

# Crea immagine di test 512x512 con pattern stellare
img = np.random.randint(20, 50, (512, 512, 3), dtype=np.uint8)

# Aggiungi stelle simulate
for i in range(50):
    x, y = np.random.randint(50, 462, 2)
    size = np.random.randint(5, 20)
    brightness = np.random.randint(200, 255)
    
    # Stella
    img[y-size:y+size, x-size:x+size] = brightness
    
Image.fromarray(img).save('test_image.jpg')
print('✅ Immagine di test creata: test_image.jpg')
"
        fi
        
        echo "🔮 Esecuzione demo..."
        python demo_enhanced_shifted.py \
            --config config_uformer.yaml \
            --input test_image.jpg \
            --output test_starless.jpg \
            --device auto
        
        echo "✅ Demo completato! Check test_starless.jpg"
        ;;
        
    "train")
        echo "🏋️ Training Mode"
        echo ""
        
        # Verifica directory dati (tiles)
        if [[ ! -d "train_tiles" ]]; then
            echo "📁 Creazione struttura directory dati tiles..."
            mkdir -p train_tiles/input
            mkdir -p train_tiles/target  
            mkdir -p val_tiles/input
            mkdir -p val_tiles/target
            
            echo "⚠️  Directory dati create ma vuote!"
            echo "Carica le tue immagini tiles in:"
            echo "  - train_tiles/input/ (tiles con stelle)"
            echo "  - train_tiles/target/ (tiles senza stelle)"
            echo "  - val_tiles/input/ (validation tiles con stelle)"
            echo "  - val_tiles/target/ (validation tiles senza stelle)"
            echo ""
        fi
        
        # Controlla se ci sono dati tiles
        train_count=$(find train_tiles/input -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
        if [[ $train_count -eq 0 ]]; then
            echo "❌ Nessun dato di training trovato!"
            echo "Carica prima le immagini tiles nelle directory"
            exit 1
        fi
        
        echo "📊 Trovate $train_count immagini tiles di training"
        echo ""
        
        # Cerca checkpoint esistente
        CHECKPOINT=""
        if [[ -f "checkpoints/best_model.pth" ]]; then
            CHECKPOINT="--pretrained checkpoints/best_model.pth"
            echo "🔄 Usando checkpoint esistente: checkpoints/best_model.pth"
        else
            echo "🆕 Training da zero (nessun checkpoint trovato)"
        fi
        
        echo "🚀 Avvio training..."
        python enhanced_uformer_finetune.py \
            --config config_uformer.yaml \
            $CHECKPOINT
        ;;
        
    "test")
        echo "🧪 Test Mode - Verifica installazione"
        echo ""
        
        echo "1️⃣ Test import moduli..."
        python -c "
from enhanced_uformer import EnhancedUFormerStarRemoval
from enhanced_loss import EnhancedUFormerLoss
import torch
print('✅ Tutti i moduli importati correttamente')
"
        
        echo "2️⃣ Test creazione modello..."
        python -c "
from enhanced_uformer import EnhancedUFormerStarRemoval
import torch

model = EnhancedUFormerStarRemoval(
    embed_dim=32,  # Molto leggero per test
    window_size=8,
    depths=[1, 1, 2, 1],  # Minimo per test
    shifted_window=True
)

print(f'✅ Modello creato: {sum(p.numel() for p in model.parameters()):,} parametri')

# Test forward pass
x = torch.randn(1, 3, 64, 64)
with torch.no_grad():
    starless, mask = model(x)
    
print(f'✅ Forward pass OK: {starless.shape}, {mask.shape}')
"
        
        echo "3️⃣ Test configurazione..."
        python -c "
import yaml
with open('config_uformer.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
print('✅ Config caricata:')
print(f'  - Win Size: {config[\"model\"][\"win_size\"]}')
print(f'  - Shifted Windows: {config[\"model\"].get(\"shifted_window\", False)}')
print(f'  - Depths: {config[\"model\"][\"depths\"]}')
"
        
        echo "✅ Tutti i test passati!"
        ;;
        
    *)
        echo "❌ Modo non riconosciuto: $MODE"
        echo ""
        echo "Uso: ./runpod_start.sh [demo|train|test]"
        echo ""
        echo "Modi disponibili:"
        echo "  demo  - Test rapido con immagine di esempio"
        echo "  train - Avvia training (richiede dati in data/)"
        echo "  test  - Verifica installazione e configurazione"
        exit 1
        ;;
esac

echo ""
echo "🎉 Operazione '$MODE' completata!"
echo ""
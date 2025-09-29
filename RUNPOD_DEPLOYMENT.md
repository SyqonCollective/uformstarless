# 🚀 RunPod Deployment Guide - Enhanced UFormer

## 📋 Setup Rapido su RunPod

### 1. 🔧 Configurazione Iniziale

**Su RunPod Terminal (root@78e930406b79:/workspace#):**

```bash
# 1. Vai nella directory workspace
cd /workspace

# 2. Scarica e esegui setup automatico
curl -sSL https://raw.githubusercontent.com/SyqonCollective/uformstarless/main/setup_runpod_enhanced.sh | bash

# OPPURE manualmente:
# wget https://raw.githubusercontent.com/SyqonCollective/uformstarless/main/setup_runpod_enhanced.sh
# chmod +x setup_runpod_enhanced.sh
# ./setup_runpod_enhanced.sh
```

**✅ Questo script:**
- Clona i file **direttamente** in `/workspace` (no sottocartelle!)
- Installa PyTorch + dipendenze
- Configura l'ambiente
- Testa il modello

### 2. 🎯 Quick Start

```bash
# Rendi eseguibile lo script di start
chmod +x runpod_start.sh

# Test rapido (crea immagine demo e la processa)
./runpod_start.sh demo

# Verifica installazione
./runpod_start.sh test

# Training (dopo aver caricato dati)
./runpod_start.sh train
```

## 📁 Struttura Directory Finale

Dopo il setup, `/workspace` conterrà:

```
/workspace/
├── enhanced_uformer.py          # Modello principale
├── config_uformer.yaml          # Config ottimizzata
├── demo_enhanced_shifted.py     # Script demo
├── enhanced_uformer_finetune.py # Training script
├── runpod_start.sh              # Quick start
├── train_tiles/                 # Training tiles dataset
│   ├── input/                  # Tiles con stelle
│   └── target/                 # Tiles senza stelle
├── val_tiles/                   # Validation tiles dataset
│   ├── input/                  # Validation tiles con stelle
│   └── target/                 # Validation tiles senza stelle
├── checkpoints/                 # Checkpoint modelli
├── experiments/                 # Risultati esperimenti
└── logs/                       # Log training
```

## 🏃‍♂️ Comandi Rapidi

### Demo Veloce
```bash
# Test con immagine auto-generata
./runpod_start.sh demo

# Test con tua immagine
python demo_enhanced_shifted.py --config config_uformer.yaml --input your_image.jpg
```

### Training

```bash
# 1. Carica i tuoi dati in data/
# 2. Training da checkpoint esistente
python enhanced_uformer_finetune.py --config config_uformer.yaml --pretrained best_model.pth

# Training da zero
python enhanced_uformer_finetune.py --config config_uformer.yaml
```

### Inference Batch
```bash
# Processa tutte le immagini in una cartella
python -c "
import glob
import subprocess

for img in glob.glob('input_images/*.jpg'):
    cmd = f'python demo_enhanced_shifted.py --config config_uformer.yaml --input {img}'
    subprocess.run(cmd, shell=True)
"
```

## ⚙️ Configurazione GPU

Il modello è **ottimizzato per A100** ma funziona su qualsiasi GPU:

```python
# Auto-detect GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Multi-GPU (se disponibili)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

**Configurazioni consigliate per GPU:**

| GPU | Batch Size | Embed Dim | Note |
|-----|-----------|-----------|------|
| A100 80GB | 32-64 | 96 | Full config |
| A100 40GB | 16-32 | 96 | Pieno potenziale |
| RTX 4090 | 8-16 | 64-96 | Ottimo |
| RTX 3090 | 4-8 | 64 | Buono |
| Altre GPU | 2-4 | 32-64 | Riduci embed_dim |

## 🔧 Troubleshooting

### Memoria GPU Insufficiente
```yaml
# In config_uformer.yaml
model:
  embed_dim: 64          # Riduci da 96
  depths: [2, 2, 4, 2]   # Riduci da [2, 2, 6, 2]

training:
  batch_size: 4          # Riduci batch size
```

### Import Errors
```bash
# Reinstalla dipendenze
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml numpy pillow tqdm
```

### Checkpoint Loading Issues
```python
# Usa strict=False per compatibilità
model.load_pretrained_compatible("checkpoint.pth", strict=False)
```

## 📊 Monitoraggio Training

### TensorBoard (opzionale)
```bash
# Installa TensorBoard
pip install tensorboard

# Avvia (se implementato nei log)
tensorboard --logdir logs --port 6006
```

### Log Checking
```bash
# Monitor training in tempo reale
tail -f logs/training.log

# Check GPU usage
nvidia-smi -l 1
```

## 🎯 Performance Tips

### 1. Compilazione PyTorch 2.0
```python
# Nel training script, abilita compilation
model = torch.compile(model, mode='max-autotune')  # Solo su PyTorch 2.0+
```

### 2. Mixed Precision
```python
# Già implementato nei training scripts
with autocast():
    output = model(input)
```

### 3. DataLoader Ottimizzato
```yaml
training:
  num_workers: 8        # Per A100, usa 4-8 workers
  pin_memory: true      # Abilita per GPU
```

## ✅ Checklist Deployment

- [ ] Script setup eseguito senza errori
- [ ] `./runpod_start.sh test` passa tutti i test
- [ ] Demo funziona: `./runpod_start.sh demo`
- [ ] Config verificata: shifted_window=true
- [ ] GPU rilevata correttamente
- [ ] Dati caricati in data/ (per training)
- [ ] Checkpoint esistente in checkpoints/ (se necessario)

## 🆘 Support

Se hai problemi:

1. **Controlla logs**: `cat logs/error.log`
2. **Test base**: `./runpod_start.sh test`
3. **GPU check**: `nvidia-smi`
4. **Dependencies**: `pip list | grep torch`

---

**🚀 Il tuo Enhanced UFormer è pronto per eliminare i quadretti 8×8 su RunPod! 🌟**
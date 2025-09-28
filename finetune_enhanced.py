"""
Fine-tuning Enhanced UFormer da checkpoint esistente
Training in 2 fasi per preservare conoscenza esistente

Fase 1: Solo nuovi moduli (Halo, Focal) - 2-3 epoche
Fase 2: Fine-tuning completo - 5-10 epoche

Comando di esecuzione:
python enhanced_uformer_finetune.py --config config_uformer.yaml --pretrained ~/Downloads/best_model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import logging
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import numpy as np

from enhanced_uformer import EnhancedUFormerStarRemoval
from star_dataset import StarRemovalDataset
from loss_uformer import UFormerLoss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Script principale per fine-tuning Enhanced UFormer
    
    Comandi:
    python enhanced_uformer_finetune.py --config config_uformer.yaml --pretrained ~/Downloads/best_model.pth
    """
    parser = argparse.ArgumentParser(description='Enhanced UFormer Fine-tuning')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--pretrained', type=str, required=True, help='Pretrained model path (best_model.pth)')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Setup Enhanced UFormer
    logger.info("Creating Enhanced UFormer model...")
    model = EnhancedUFormerStarRemoval(
        embed_dim=config['model']['embed_dim'],
        window_size=16,  # Aumentato da 8 per ridurre quadretti  
        halo_size=8,     # Halo per vedere oltre finestre
        depths=config['model']['depths'],
        num_heads=config['model']['num_heads'],
        focal_interval=2,  # Focal block ogni 2 blocchi
    ).to(device)
    
    # Load pretrained checkpoint (compatibile)
    logger.info(f"Loading pretrained model from {args.pretrained}")
    try:
        missing_keys, unexpected_keys = model.load_pretrained_compatible(
            args.pretrained, strict=False
        )
        logger.info(f"Loaded pretrained model - Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
    except Exception as e:
        logger.error(f"Failed to load pretrained model: {e}")
        return
    
    # Setup loss
    criterion = UFormerLoss(mask_weight=config.get('mask_weight', 0.1))
    
    # Setup datasets
    logger.info("Setting up datasets...")
    train_dataset = StarRemovalDataset(
        input_dir=config['data']['train_input_dir'],
        target_dir=config['data']['train_target_dir'], 
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    val_dataset = StarRemovalDataset(
        input_dir=config['data']['val_input_dir'],
        target_dir=config['data']['val_target_dir'],
        augment=False
    )
    
    # Validation subset per velocità
    val_subset_size = config['training'].get('val_subset_size', 1000)
    if len(val_dataset) > val_subset_size:
        indices = np.random.choice(len(val_dataset), val_subset_size, replace=False)
        val_dataset = torch.utils.data.Subset(val_dataset, indices)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['val_batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Setup experiment directory
    exp_name = f"enhanced_uformer_finetune_{int(time.time())}"
    exp_dir = Path(config['experiment']['output_dir']) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = exp_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    logger.info(f"Experiment directory: {exp_dir}")
    
    # Training functions
    def train_epoch(model, train_loader, criterion, optimizer, epoch, device, phase_name="Training"):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"{phase_name} Epoch {epoch}")
        
        for batch_idx, (inputs, targets, masks) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_starless, pred_mask = model(inputs)
            
            # Compute loss
            loss, img_loss, mask_loss = criterion(pred_starless, pred_mask, targets, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({'Loss': f'{total_loss/(batch_idx+1):.4f}'})
        
        return total_loss / len(train_loader)
    
    def validate(model, val_loader, criterion, device, epoch):
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, masks in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                masks = masks.to(device)
                
                pred_starless, pred_mask = model(inputs)
                loss, _, _ = criterion(pred_starless, pred_mask, targets, masks)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir, phase, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 
            'loss': loss,
            'phase': phase
        }
        
        checkpoint_path = checkpoint_dir / f'enhanced_uformer_{phase}_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = checkpoint_dir / f'enhanced_best_model_{phase}.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
    
    # FASE 1: Solo nuovi moduli (Halo, Focal)
    logger.info("=== PHASE 1: Training only new Halo/Focal modules ===")
    
    # Freeze pretrained modules
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if 'halo' in name.lower() or 'focal' in name.lower():
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    logger.info(f"Phase 1 - Frozen: {frozen_count:,}, Trainable: {trainable_count:,}")
    
    # Optimizer per fase 1
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_phase1 = optim.AdamW(
        trainable_params,
        lr=config['training']['optimizer']['lr'] * 0.1,  # LR ridotto
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase1, T_max=3, eta_min=1e-6)
    
    best_val_loss_phase1 = float('inf')
    
    for epoch in range(1, 4):  # 3 epoche
        train_loss = train_epoch(model, train_loader, criterion, optimizer_phase1, epoch, device, "Phase1")
        val_loss = validate(model, val_loader, criterion, device, epoch)
        
        scheduler_phase1.step()
        
        logger.info(f"Phase 1 Epoch {epoch} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        is_best = val_loss < best_val_loss_phase1
        if is_best:
            best_val_loss_phase1 = val_loss
        
        save_checkpoint(model, optimizer_phase1, epoch, val_loss, checkpoint_dir, "phase1", is_best)
    
    logger.info(f"Phase 1 completed. Best val loss: {best_val_loss_phase1:.4f}")
    
    # FASE 2: Fine-tuning completo
    logger.info("=== PHASE 2: Full model fine-tuning ===")
    
    # Unfreeze all modules
    for param in model.parameters():
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Phase 2 - All trainable parameters: {total_params:,}")
    
    # Optimizer per fase 2 con LR molto basso
    optimizer_phase2 = optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'] * 0.01,  # LR molto ridotto
        weight_decay=config['training']['optimizer']['weight_decay']
    )
    
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_phase2, T_max=10, eta_min=1e-7)
    
    best_val_loss_phase2 = float('inf')
    
    for epoch in range(1, 11):  # 10 epoche
        train_loss = train_epoch(model, train_loader, criterion, optimizer_phase2, epoch, device, "Phase2")
        val_loss = validate(model, val_loader, criterion, device, epoch)
        
        scheduler_phase2.step()
        
        logger.info(f"Phase 2 Epoch {epoch} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        is_best = val_loss < best_val_loss_phase2
        if is_best:
            best_val_loss_phase2 = val_loss
        
        save_checkpoint(model, optimizer_phase2, epoch, val_loss, checkpoint_dir, "phase2", is_best)
    
    logger.info(f"Phase 2 completed. Best val loss: {best_val_loss_phase2:.4f}")
    logger.info(f"Training completed! Best models saved in: {checkpoint_dir}")


if __name__ == '__main__':
    main()

"""
Fine-tuning COMPATIBILE del UFormer esistente
Aggiunge solo Halo Attention mantenendo 100% compatibilità
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse
import yaml
import os
import logging
from tqdm import tqdm
import time

# Local imports
from star_dataset import StarRemovalDataset
from compatible_enhanced_uformer import CompatibleEnhancedUFormer
from loss_uformer import UFormerLoss


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('compatible_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_and_load_checkpoint(config, checkpoint_path, device):
    """
    Crea CompatibleEnhancedUFormer e carica checkpoint esistente
    """
    logger = logging.getLogger(__name__)
    
    # Crea modello Enhanced COMPATIBILE
    model = CompatibleEnhancedUFormer(
        embed_dim=config.get('embed_dim', 96),
        window_size=config.get('window_size', 8),
        halo_size=config.get('halo_size', 4),  # Halo piccolo per compatibilità
        depths=config.get('depths', [2, 2, 6, 2]),
        num_heads=config.get('num_heads', [3, 6, 12, 24]),
        mlp_ratio=config.get('mlp_ratio', 4.0),
        patch_size=config.get('patch_size', 4)
    ).to(device)
    
    logger.info(f"Created CompatibleEnhancedUFormer model")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Carica checkpoint esistente - DEVE essere 100% compatibile
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading compatible checkpoint from {checkpoint_path}")
        missing, unexpected = model.load_pretrained_compatible(checkpoint_path)
        
        if not missing and not unexpected:
            logger.info("✓ Checkpoint loaded successfully - FULL COMPATIBILITY")
        else:
            logger.error("✗ Checkpoint not fully compatible!")
            raise RuntimeError("Checkpoint incompatible!")
    else:
        logger.warning("No checkpoint provided - training from scratch")
    
    return model


def fine_tune_compatible(model, train_loader, val_loader, config, device, logger):
    """
    Fine-tuning GRADUALE per Halo Attention
    
    Fase 1: Warm-up Halo Attention (pochi epochs, LR basso)
    Fase 2: Fine-tuning completo (LR molto basso)
    """
    
    # Loss function
    criterion = UFormerLoss(
        l1_weight=config.get('l1_weight', 1.0),
        perceptual_weight=config.get('perceptual_weight', 0.1),
        mask_weight=config.get('mask_weight', 0.5)
    ).to(device)
    
    # Optimizer con LR molto basso per fine-tuning
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get('learning_rate', 1e-5),  # LR molto basso
        weight_decay=config.get('weight_decay', 1e-4),
        betas=(0.9, 0.999)
    )
    
    # Scheduler più graduale
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config.get('epochs', 20),
        eta_min=config.get('min_lr', 1e-6)
    )
    
    # Training loop
    best_loss = float('inf')
    epochs = config.get('epochs', 20)
    
    logger.info(f"Starting compatible fine-tuning for {epochs} epochs")
    logger.info(f"Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (input_imgs, target_imgs) in enumerate(pbar):
            input_imgs = input_imgs.to(device, non_blocking=True)
            target_imgs = target_imgs.to(device, non_blocking=True)
            
            # Forward pass
            optimizer.zero_grad()
            pred_starless, pred_mask = model(input_imgs)
            
            # Loss calculation
            loss = criterion(pred_starless, pred_mask, target_imgs, input_imgs)
            
            # Backward pass con gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Clip più conservativo
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            # Update progress
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{train_loss/train_batches:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Memory cleanup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for input_imgs, target_imgs in pbar_val:
                input_imgs = input_imgs.to(device, non_blocking=True)
                target_imgs = target_imgs.to(device, non_blocking=True)
                
                pred_starless, pred_mask = model(input_imgs)
                loss = criterion(pred_starless, pred_mask, target_imgs, input_imgs)
                
                val_loss += loss.item()
                val_batches += 1
                
                pbar_val.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{val_loss/val_batches:.4f}'
                })
        
        avg_val_loss = val_loss / val_batches
        
        # Update scheduler
        scheduler.step()
        
        # Log progress
        logger.info(f"Epoch {epoch+1} Complete:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        logger.info(f"  Val Loss:   {avg_val_loss:.4f}")
        logger.info(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }
            
            os.makedirs('experiments/compatible_enhanced/checkpoints', exist_ok=True)
            torch.save(checkpoint, 'experiments/compatible_enhanced/checkpoints/best_model.pth')
            logger.info(f"  ✓ New best model saved (val_loss: {best_loss:.4f})")
        
        # Memory cleanup
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Compatible Enhanced UFormer Fine-tuning')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--pretrained', type=str, required=True, help='Pretrained model path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Create datasets
    logger.info("Creating datasets...")
    
    train_dataset = StarRemovalDataset(
        input_dir=config['data']['train_input_dir'],
        target_dir=config['data']['train_target_dir'],
        image_size=config['data']['image_size'],
        augment=config['data'].get('augment', True)
    )
    
    val_dataset = StarRemovalDataset(
        input_dir=config['data']['val_input_dir'],
        target_dir=config['data']['val_target_dir'],
        image_size=config['data']['image_size'],
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} samples")
    logger.info(f"Val dataset: {len(val_dataset)} samples")
    
    # Create model and load checkpoint
    model = create_model_and_load_checkpoint(config, args.pretrained, device)
    
    # Fine-tune model
    logger.info("Starting compatible fine-tuning...")
    fine_tune_compatible(model, train_loader, val_loader, config['training'], device, logger)
    
    logger.info("Compatible fine-tuning completed!")


if __name__ == '__main__':
    main()

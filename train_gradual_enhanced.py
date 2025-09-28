"""
Training GRADUALE per Enhanced UFormer
Approccio: Carica checkpoint esistente -> Converti a Halo -> Fine-tune
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
import os
import logging
from tqdm import tqdm

# Local imports
from star_dataset import StarRemovalDataset
from gradual_enhanced_uformer import create_gradually_enhanced_model
from loss_uformer import UFormerLoss


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gradual_enhancement_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def gradual_fine_tuning(model, train_loader, val_loader, config, device, logger):
    """
    Fine-tuning GRADUALE in 3 fasi:
    
    Fase 1: Warm-up Halo (2 epochs, LR alto, solo nuovi parametri)
    Fase 2: Gentle tuning (5 epochs, LR medio, tutti i parametri)
    Fase 3: Fine tuning (10 epochs, LR basso, tutti i parametri)
    """
    
    # Loss function
    criterion = UFormerLoss(
        l1_weight=config.get('l1_weight', 1.0),
        perceptual_weight=config.get('perceptual_weight', 0.1),
        mask_weight=config.get('mask_weight', 0.5)
    ).to(device)
    
    best_loss = float('inf')
    total_epochs = 0
    
    # FASE 1: Warm-up Halo (solo parametri Halo, se fossero separabili)
    logger.info("=" * 50)
    logger.info("FASE 1: Halo Warm-up (2 epochs)")
    logger.info("=" * 50)
    
    # Tutti i parametri ma LR molto basso per iniziare gentilmente
    optimizer_phase1 = optim.AdamW(
        model.parameters(),
        lr=5e-6,  # LR molto basso
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    for epoch in range(2):
        total_epochs += 1
        logger.info(f"Phase 1 - Epoch {epoch+1}/2 (Total: {total_epochs})")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer_phase1, criterion, device, f"P1E{epoch+1}")
        val_loss = validate_epoch(model, val_loader, criterion, device, f"P1E{epoch+1}")
        
        logger.info(f"Phase 1 Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer_phase1, total_epochs, train_loss, val_loss, config, "phase1_best")
            logger.info(f"  ✓ New best model saved")
    
    # FASE 2: Gentle tuning
    logger.info("=" * 50)
    logger.info("FASE 2: Gentle Tuning (5 epochs)")
    logger.info("=" * 50)
    
    optimizer_phase2 = optim.AdamW(
        model.parameters(),
        lr=2e-5,  # LR medio
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2, T_max=5, eta_min=1e-6
    )
    
    for epoch in range(5):
        total_epochs += 1
        logger.info(f"Phase 2 - Epoch {epoch+1}/5 (Total: {total_epochs})")
        
        train_loss = train_epoch(model, train_loader, optimizer_phase2, criterion, device, f"P2E{epoch+1}")
        val_loss = validate_epoch(model, val_loader, criterion, device, f"P2E{epoch+1}")
        
        scheduler_phase2.step()
        
        logger.info(f"Phase 2 Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={optimizer_phase2.param_groups[0]['lr']:.2e}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer_phase2, total_epochs, train_loss, val_loss, config, "phase2_best")
            logger.info(f"  ✓ New best model saved")
    
    # FASE 3: Fine tuning  
    logger.info("=" * 50)
    logger.info("FASE 3: Fine Tuning (10 epochs)")
    logger.info("=" * 50)
    
    optimizer_phase3 = optim.AdamW(
        model.parameters(),
        lr=1e-5,  # LR basso
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    scheduler_phase3 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase3, T_max=10, eta_min=5e-7
    )
    
    for epoch in range(10):
        total_epochs += 1
        logger.info(f"Phase 3 - Epoch {epoch+1}/10 (Total: {total_epochs})")
        
        train_loss = train_epoch(model, train_loader, optimizer_phase3, criterion, device, f"P3E{epoch+1}")
        val_loss = validate_epoch(model, val_loader, criterion, device, f"P3E{epoch+1}")
        
        scheduler_phase3.step()
        
        logger.info(f"Phase 3 Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, LR={optimizer_phase3.param_groups[0]['lr']:.2e}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer_phase3, total_epochs, train_loss, val_loss, config, "final_best")
            logger.info(f"  ✓ New best model saved")
    
    logger.info("=" * 50)
    logger.info(f"Gradual training completed!")
    logger.info(f"Total epochs: {total_epochs}")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    logger.info("=" * 50)


def train_epoch(model, train_loader, optimizer, criterion, device, phase_name):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Training {phase_name}")
    for batch_idx, (input_imgs, target_imgs) in enumerate(pbar):
        input_imgs = input_imgs.to(device, non_blocking=True)
        target_imgs = target_imgs.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        pred_starless, pred_mask = model(input_imgs)
        
        # Loss calculation
        loss = criterion(pred_starless, pred_mask, target_imgs, input_imgs)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg': f'{total_loss/num_batches:.4f}'
        })
        
        # Memory cleanup
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    return total_loss / num_batches


def validate_epoch(model, val_loader, criterion, device, phase_name):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validation {phase_name}")
        for input_imgs, target_imgs in pbar:
            input_imgs = input_imgs.to(device, non_blocking=True)
            target_imgs = target_imgs.to(device, non_blocking=True)
            
            pred_starless, pred_mask = model(input_imgs)
            loss = criterion(pred_starless, pred_mask, target_imgs, input_imgs)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/num_batches:.4f}'
            })
    
    return total_loss / num_batches


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, suffix):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }
    
    os.makedirs('experiments/gradual_enhanced/checkpoints', exist_ok=True)
    checkpoint_path = f'experiments/gradual_enhanced/checkpoints/{suffix}.pth'
    torch.save(checkpoint, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description='Gradual Enhanced UFormer Training')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--pretrained', type=str, required=True, help='Original pretrained model path')
    parser.add_argument('--halo-size', type=int, default=4, help='Halo size for attention')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Halo size: {args.halo_size}")
    
    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Create enhanced model from existing checkpoint
    logger.info("Creating gradually enhanced model...")
    model = create_gradually_enhanced_model(
        checkpoint_path=args.pretrained,
        halo_size=args.halo_size,
        device=device
    )
    
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
    
    # Start gradual fine-tuning
    logger.info("Starting gradual enhancement training...")
    gradual_fine_tuning(model, train_loader, val_loader, config['training'], device, logger)
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()

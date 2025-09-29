#!/usr/bin/env python3
"""
Training script modificato per preservare dettagli
Usa StarFocusedLoss che modifica SOLO le stelle
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from enhanced_uformer import EnhancedUFormerStarRemoval
from star_dataset import create_dataloader
from star_focused_loss import StarFocusedLoss, DetailPreservingLoss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DetailPreservingTrainer:
    """Trainer che preserva dettagli modificando solo stelle"""
    
    def __init__(self, config_path: str, pretrained_path: str = None):
        self.config_path = config_path
        self.pretrained_path = pretrained_path
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Setup model
        self.setup_model()
        
        # Setup loss - NUOVA LOSS STAR-FOCUSED
        self.criterion = StarFocusedLoss(
            l1_weight=1.0,
            perceptual_weight=0.0,      # DISABILITATO
            mask_focus_weight=5.0,      # Focus su stelle
            preserve_weight=10.0        # Preserva resto
        ).to(self.device)
        
        # Setup datasets
        self.setup_datasets()
        
        # Setup experiment dir
        self.setup_experiment_dir()
        
    def setup_model(self):
        """Setup modello Enhanced UFormer"""
        self.model = EnhancedUFormerStarRemoval(
            embed_dim=self.config['model']['embed_dim'],
            window_size=self.config['model']['win_size'],
            halo_size=self.config['model']['halo_size'],
            depths=self.config['model']['depths'],
            num_heads=self.config['model']['num_heads'],
            focal_interval=self.config['model'].get('focal_interval', 999),
            shifted_window=self.config['model'].get('shifted_window', True)
        ).to(self.device)
        
        # Carica pretrained se specificato
        if self.pretrained_path:
            self.load_pretrained()
            
    def load_pretrained(self):
        """Carica checkpoint esistente"""
        if not self.pretrained_path or not Path(self.pretrained_path).exists():
            logger.warning(f"Checkpoint non trovato: {self.pretrained_path}")
            return
            
        logger.info(f"Loading pretrained model from {self.pretrained_path}")
        
        checkpoint = torch.load(self.pretrained_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded with {len(missing_keys)} missing keys")
        
    def setup_datasets(self):
        """Setup datasets"""
        # Training loader
        self.train_loader = create_dataloader(
            input_dir=self.config['data']['train_input_dir'],
            target_dir=self.config['data']['train_target_dir'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers'],
            shuffle=True,
            is_training=True
        )
        
        # Validation loader
        self.val_loader = create_dataloader(
            input_dir=self.config['data']['val_input_dir'],
            target_dir=self.config['data']['val_target_dir'],
            batch_size=self.config['training'].get('val_batch_size', 8),
            num_workers=self.config['training']['num_workers'],
            shuffle=False,
            is_training=False
        )
        
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
    def setup_experiment_dir(self):
        """Setup directory esperimento"""
        self.experiment_dir = Path(self.config['experiment']['output_dir']) / self.config['experiment']['name']
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment directory: {self.experiment_dir}")
        
    def train_epoch(self, optimizer, epoch):
        """Training epoch con nuova loss"""
        self.model.train()
        total_loss = 0.0
        total_components = {}
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)      # Original image
            targets = batch['target'].to(self.device)    # Starless target  
            masks = batch['mask'].to(self.device)        # Star mask
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_starless, pred_mask = self.model(inputs)
            
            # NUOVA LOSS - include input originale
            loss, loss_dict = self.criterion(
                pred_starless, pred_mask, targets, masks, inputs  # +inputs!
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in total_components:
                    total_components[key] = 0
                total_components[key] += value
                
            # Update progress bar
            if batch_idx % 10 == 0:
                num_batches = batch_idx + 1
                avg_loss = total_loss / num_batches
                avg_star = total_components.get("star_area_loss", 0) / num_batches
                avg_preserve = total_components.get("preserve_loss", 0) / num_batches
                
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Star': f'{avg_star:.4f}',
                    'Preserve': f'{avg_preserve:.4f}'
                })
                
        # Calculate averages
        avg_loss = total_loss / len(self.train_loader)
        avg_components = {k: v / len(self.train_loader) for k, v in total_components.items()}
        
        return avg_loss, avg_components
        
    def validate(self, epoch):
        """Validation epoch"""
        self.model.eval()
        total_loss = 0.0
        total_components = {}
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                pred_starless, pred_mask = self.model(inputs)
                loss, loss_dict = self.criterion(pred_starless, pred_mask, targets, masks, inputs)
                
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    if key not in total_components:
                        total_components[key] = 0
                    total_components[key] += value
                    
        avg_loss = total_loss / len(self.val_loader)
        avg_components = {k: v / len(self.val_loader) for k, v in total_components.items()}
        
        return avg_loss, avg_components
        
    def run_training(self):
        """Run complete training loop"""
        # Setup optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['optimizer']['lr'],
            weight_decay=self.config['training']['optimizer']['weight_decay']
        )
        
        # Setup scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=self.config['training']['scheduler']['eta_min']
        )
        
        best_val_loss = float('inf')
        
        logger.info("Starting Detail-Preserving UFormer training")
        
        try:
            for epoch in range(1, self.config['training']['epochs'] + 1):
                # Training
                train_loss, train_components = self.train_epoch(optimizer, epoch)
                
                # Validation
                val_loss, val_components = self.validate(epoch)
                
                # Scheduler step
                scheduler.step()
                
                # Logging
                logger.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                logger.info(f"Train Components: {train_components}")
                logger.info(f"Val Components: {val_components}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, self.experiment_dir / 'best_detail_preserving_model.pth')
                    logger.info(f"Saved best model at epoch {epoch}")
                    
                # Save checkpoint
                if epoch % self.config['training']['save_interval'] == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                    }, self.experiment_dir / f'checkpoint_epoch_{epoch:04d}.pth')
                    
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Detail-Preserving UFormer Training')
    parser.add_argument('--config', type=str, default='config_detail_preserving.yaml',
                       help='Config file path')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Pretrained model path')
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = DetailPreservingTrainer(args.config, args.pretrained)
    trainer.run_training()


if __name__ == '__main__':
    main()
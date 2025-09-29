"""
Enhanced UFormer Training Script - Corretto
Training script per Enhanced UFormer con shifted windows e perceptual loss
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
from star_dataset import StarDataset
from enhanced_loss import EnhancedUFormerLoss

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedUFormerTrainer:
    """
    Trainer per Enhanced UFormer con shifted windows
    """
    
    def __init__(self, config_path: str, pretrained_path: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_path = pretrained_path
        
        # Setup modello Enhanced
        self.model = EnhancedUFormerStarRemoval(
            embed_dim=self.config['model']['embed_dim'],
            window_size=self.config['model']['win_size'],
            halo_size=self.config['model'].get('halo_size', 4),
            depths=self.config['model']['depths'],
            num_heads=self.config['model']['num_heads'],
            focal_interval=self.config['model'].get('focal_interval', 2),
            shifted_window=self.config['model'].get('shifted_window', True)
        ).to(self.device)
        
        # Carica checkpoint esistente se specificato
        if self.pretrained_path:
            self.load_pretrained()
        
        # Setup Enhanced loss con Perceptual Loss
        loss_config = self.config.get('loss', {})
        self.criterion = EnhancedUFormerLoss(
            l1_weight=loss_config.get('l1_weight', 1.0),
            perceptual_weight=loss_config.get('perceptual_weight', 0.1),
            ssim_weight=loss_config.get('ssim_weight', 0.1),
            mask_weight=loss_config.get('mask_weight', 0.1),
            use_ssim=loss_config.get('use_ssim', True)
        )
        
        # Setup datasets
        self.setup_datasets()
        
        # Setup experiment directory
        self.setup_experiment_dir()
        
    def load_pretrained(self):
        """Carica checkpoint esistente in modo compatibile"""
        if not self.pretrained_path or not Path(self.pretrained_path).exists():
            logger.warning(f"Checkpoint non trovato: {self.pretrained_path}")
            return
            
        logger.info(f"Loading pretrained model from {self.pretrained_path}")
        
        missing_keys, unexpected_keys = self.model.load_pretrained_compatible(
            self.pretrained_path, strict=False
        )
        
        logger.info(f"Successfully loaded pretrained model")
        logger.info(f"New modules (will be trained from scratch): {len(missing_keys)}")
        
        if missing_keys:
            logger.info(f"Example missing keys: {missing_keys[:5]}")
    
    def setup_datasets(self):
        """Setup training e validation datasets"""
        train_config = self.config['data']
        
        # Training dataset
        self.train_dataset = StarDataset(
            input_dir=train_config['train_input_dir'],
            target_dir=train_config['train_target_dir'],
            transform=True,
            cache_size=1000
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True,
            drop_last=True
        )
        
        # Validation dataset
        self.val_dataset = StarDataset(
            input_dir=train_config['val_input_dir'],
            target_dir=train_config['val_target_dir'],
            transform=False,
            cache_size=500
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training'].get('val_batch_size', 8),
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True
        )
        
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def setup_experiment_dir(self):
        """Setup directory esperimenti"""
        exp_name = self.config.get('experiment', {}).get('name', 'enhanced_uformer')
        self.checkpoint_dir = Path(f"experiments/{exp_name}")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment directory: {self.checkpoint_dir}")
    
    def train_epoch(self, optimizer, epoch):
        """Training per una epoca"""
        self.model.train()
        total_loss = 0.0
        total_components = {}
        
        pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, (inputs, targets, masks) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            if masks is not None:
                masks = masks.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_starless, pred_mask = self.model(inputs)
            
            # Compute enhanced loss
            loss, loss_dict = self.criterion(pred_starless, pred_mask, targets, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in total_components:
                    total_components[key] = 0
                total_components[key] += value
            
            # Update progress bar
            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'L1': f'{total_components.get("l1_loss", 0)/(batch_idx+1):.4f}',
                    'Perc': f'{total_components.get("perceptual_loss", 0)/(batch_idx+1):.4f}'
                })
        
        # Calcola medie
        avg_loss = total_loss / len(self.train_loader)
        avg_components = {k: v / len(self.train_loader) for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def validate(self, epoch):
        """Validazione"""
        self.model.eval()
        total_loss = 0.0
        total_components = {}
        
        with torch.no_grad():
            for inputs, targets, masks in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                if masks is not None:
                    masks = masks.to(self.device)
                
                pred_starless, pred_mask = self.model(inputs)
                loss, loss_dict = self.criterion(pred_starless, pred_mask, targets, masks)
                
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    if key not in total_components:
                        total_components[key] = 0
                    total_components[key] += value
        
        avg_loss = total_loss / len(self.val_loader)
        avg_components = {k: v / len(self.val_loader) for k, v in total_components.items()}
        
        return avg_loss, avg_components
    
    def save_checkpoint(self, epoch, optimizer, loss, is_best=False):
        """Salva checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Checkpoint corrente
        checkpoint_path = self.checkpoint_dir / f'enhanced_uformer_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_enhanced_model.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved: {best_path}")
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def run_training(self):
        """Esegue training completo"""
        logger.info("Starting Enhanced UFormer training")
        start_time = time.time()
        
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
        
        try:
            for epoch in range(1, self.config['training']['epochs'] + 1):
                # Training
                train_loss, train_components = self.train_epoch(optimizer, epoch)
                
                # Validation
                val_loss, val_components = self.validate(epoch)
                
                # Scheduler step
                scheduler.step()
                
                # Logging
                logger.info(f"Epoch {epoch}/{self.config['training']['epochs']}")
                logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                logger.info(f"Train L1: {train_components.get('l1_loss', 0):.4f} | "
                          f"Perceptual: {train_components.get('perceptual_loss', 0):.4f}")
                
                # Save checkpoint
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                
                if epoch % self.config['training'].get('save_interval', 5) == 0 or is_best:
                    self.save_checkpoint(epoch, optimizer, val_loss, is_best)
            
            # Final summary
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time/3600:.2f} hours")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
            logger.info(f"Models saved in: {self.checkpoint_dir}")
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Enhanced UFormer Training')
    parser.add_argument('--config', type=str, default='config_uformer.yaml', 
                       help='Config file path')
    parser.add_argument('--pretrained', type=str, default=None,
                       help='Pretrained model path')
    
    args = parser.parse_args()
    
    # Crea trainer e avvia training
    trainer = EnhancedUFormerTrainer(args.config, args.pretrained)
    trainer.run_training()


if __name__ == '__main__':
    main()
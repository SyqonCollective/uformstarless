"""
Training script for Quick Fix UFormer - starts from existing checkpoint
Continues training with shifted window architecture (no quadretti!)
"""

import os
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from quick_fix_uformer import QuickFixUFormerStarRemoval, copy_compatible_weights
from star_dataset import create_dataloader
from loss_uformer import UFormerLoss, UFormerMetrics


class QuickFixTrainer:
    """
    Quick Fix UFormer trainer - continues from existing UFormer checkpoint
    Uses shifted window architecture to eliminate quadretti artifacts
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize Quick Fix model
        self.model = self.build_quick_fix_model()
        
        # Load weights from existing checkpoint
        self.load_pretrained_weights()
        
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.criterion = UFormerLoss(mask_weight=config.get('mask_weight', 0.1))
        
        # A100 optimizations
        self.scaler = GradScaler()
        
        # Setup data loaders
        self.train_loader = create_dataloader(
            config['data']['train_input_dir'],
            config['data']['train_target_dir'], 
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            shuffle=True,
            is_training=True
        )
        
        self.val_loader = create_dataloader(
            config['data']['val_input_dir'],
            config['data']['val_target_dir'],
            batch_size=config['training']['val_batch_size'],
            num_workers=config['training']['num_workers'],
            shuffle=False,
            is_training=False
        )
        
        # Setup experiment tracking
        self.setup_experiment_tracking()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['experiment']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'quick_fix_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def build_quick_fix_model(self):
        """Build Quick Fix UFormer model"""
        model = QuickFixUFormerStarRemoval(
            embed_dim=self.config['model']['embed_dim'],
            window_size=self.config['model']['window_size'],
            depths=self.config['model']['depths'],
            num_heads=self.config['model']['num_heads']
        ).to(self.device)
        
        self.logger.info(f"✨ Built Quick Fix UFormer model (eliminates quadretti!)")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    def load_pretrained_weights(self):
        """Load weights from existing UFormer checkpoint"""
        pretrained_path = self.config.get('pretrained_checkpoint')
        if not pretrained_path or not Path(pretrained_path).exists():
            self.logger.warning("No pretrained checkpoint specified or found!")
            return
        
        try:
            self.logger.info(f"🔄 Loading pretrained weights from {pretrained_path}")
            
            # Load original checkpoint
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                pretrained_state = checkpoint['model_state_dict']
                self.current_epoch = checkpoint.get('epoch', 0)
                self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                self.logger.info(f"📊 Starting from epoch {self.current_epoch}, best loss: {self.best_val_loss:.6f}")
            else:
                pretrained_state = checkpoint
                self.logger.info("Loading direct state dict (no training info)")
            
            # Copy compatible weights to Quick Fix model
            copied_layers = copy_compatible_weights(self.model, pretrained_state)
            
            self.logger.info(f"✅ Transferred weights from {copied_layers} compatible layers")
            self.logger.info("🎯 Quick Fix model ready - will eliminate quadretti!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load pretrained weights: {e}")
            raise
    
    def build_optimizer(self):
        """Build optimizer"""
        if self.config['training']['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate']
            )
        
        self.logger.info(f"Built {self.config['training']['optimizer']} optimizer")
        return optimizer
    
    def build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config['training']['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=self.config['training']['learning_rate'] * 0.01
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['scheduler_step'],
                gamma=self.config['training']['scheduler_gamma']
            )
        
        return scheduler
    
    def setup_experiment_tracking(self):
        """Setup TensorBoard and checkpoints"""
        self.experiment_dir = Path(self.config['experiment']['output_dir'])
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.tensorboard_dir = self.experiment_dir / 'tensorboard'
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.tensorboard_dir)
        
        # Save config
        config_path = self.experiment_dir / 'quick_fix_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        self.logger.info(f"📁 Experiment directory: {self.experiment_dir}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        metrics = UFormerMetrics()
        
        pbar = tqdm(self.train_loader, desc=f"Quick Fix Epoch {self.current_epoch}")
        
        for batch_idx, (input_imgs, target_imgs, masks) in enumerate(pbar):
            input_imgs = input_imgs.to(self.device, non_blocking=True)
            target_imgs = target_imgs.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            with autocast():
                starless_pred, mask_pred = self.model(input_imgs)
                loss_dict = self.criterion(starless_pred, target_imgs, mask_pred, masks)
                total_loss = loss_dict['total_loss']
            
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            metrics.update(loss_dict, starless_pred, target_imgs, mask_pred, masks)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss.item():.6f}",
                'psnr': f"{metrics.get_avg_psnr():.2f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to tensorboard
            if self.global_step % self.config['training']['log_interval'] == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f'Train/{key}', value.item(), self.global_step)
                self.writer.add_scalar('Train/PSNR', metrics.get_avg_psnr(), self.global_step)
                self.writer.add_scalar('Train/LR', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        return metrics.get_epoch_summary()
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        metrics = UFormerMetrics()
        
        with torch.no_grad():
            for input_imgs, target_imgs, masks in tqdm(self.val_loader, desc="Validation"):
                input_imgs = input_imgs.to(self.device, non_blocking=True)
                target_imgs = target_imgs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)
                
                with autocast():
                    starless_pred, mask_pred = self.model(input_imgs)
                    loss_dict = self.criterion(starless_pred, target_imgs, mask_pred, masks)
                
                metrics.update(loss_dict, starless_pred, target_imgs, mask_pred, masks)
        
        return metrics.get_epoch_summary()
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'quick_fix_epoch_{self.current_epoch:04d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Best model
        if is_best:
            best_path = self.checkpoint_dir / 'quick_fix_best.pth'
            torch.save(checkpoint, best_path)
            
            # Also save just the model for inference
            model_only_path = self.checkpoint_dir / 'quick_fix_model_only.pth'
            torch.save(self.model.state_dict(), model_only_path)
            
            self.logger.info(f"💾 Saved best Quick Fix model: {best_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("🚀 Starting Quick Fix UFormer training (no quadretti!)")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Starting epoch: {self.current_epoch}")
        self.logger.info(f"Total epochs: {self.config['training']['epochs']}")
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.6f}, "
                           f"Val Loss: {val_metrics['total_loss']:.6f}, "
                           f"PSNR: {val_metrics['psnr']:.2f}")
            
            # Log to tensorboard
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Save checkpoint
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
            
            self.save_checkpoint(is_best)
            
            # Early stopping check
            if self.should_early_stop(val_metrics):
                self.logger.info("Early stopping triggered")
                break
        
        self.writer.close()
        self.logger.info("✅ Quick Fix training completed!")
    
    def should_early_stop(self, val_metrics: Dict[str, float]) -> bool:
        """Check if early stopping should be triggered"""
        patience = self.config['training'].get('early_stopping_patience')
        if patience is None:
            return False
        
        # Implementation would track validation loss history
        # For now, return False (no early stopping)
        return False


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick Fix UFormer Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained UFormer checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add pretrained checkpoint to config
    config['pretrained_checkpoint'] = args.pretrained
    
    # Initialize and start training
    trainer = QuickFixTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()

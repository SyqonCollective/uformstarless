"""
A100 SXM optimized training loop for UFormer star removal
Designed for maximum efficiency and speed on A100 80GB GPU
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

from uformer import UFormerStarRemoval
from star_dataset import create_dataloader
from loss_uformer import UFormerLoss, UFormerMetrics


class A100Trainer:
    """
    A100 SXM optimized trainer for UFormer star removal
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model, optimizer, loss
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.criterion = UFormerLoss(mask_weight=config.get('mask_weight', 0.1))
        
        # A100 optimizations
        self.scaler = GradScaler()  # Mixed precision
        self.compile_model()  # PyTorch 2.0 compilation
        
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
        
        # Use subset for faster validation if specified
        val_subset_size = config['training'].get('val_subset_size')
        if val_subset_size and val_subset_size < len(self.val_loader.dataset):
            indices = list(range(val_subset_size))
            val_subset = torch.utils.data.Subset(self.val_loader.dataset, indices)
            self.val_loader = torch.utils.data.DataLoader(
                val_subset,
                batch_size=config['training']['val_batch_size'],
                num_workers=config['training']['num_workers'],
                shuffle=False,
                pin_memory=True
            )
        
        # Setup experiment tracking
        self.setup_experiment_tracking()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        # Resume from checkpoint if specified
        self.resume_from_checkpoint()
        
    def resume_from_checkpoint(self):
        """Resume training from checkpoint if specified in config"""
        resume_path = self.config.get('resume_checkpoint')
        if resume_path and Path(resume_path).exists():
            self.logger.info(f"Resuming training from {resume_path}")
            try:
                checkpoint = torch.load(resume_path, map_location=self.device)
                
                # Load model state
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Load optimizer state
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Load scheduler state
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                # Load training state
                self.current_epoch = checkpoint.get('epoch', 0)
                self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                
                self.logger.info(f"Resumed from epoch {self.current_epoch}, best val loss: {self.best_val_loss:.6f}")
                
            except Exception as e:
                self.logger.error(f"Failed to load checkpoint: {e}")
                self.logger.info("Starting training from scratch")
        elif resume_path:
            self.logger.warning(f"Checkpoint file not found: {resume_path}")
            self.logger.info("Starting training from scratch")
        
    def setup_logging(self):
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def build_model(self) -> nn.Module:
        """Build UFormer model"""
        model_config = self.config['model']
        model = UFormerStarRemoval(
            in_channels=3,
            out_channels=3,
            embed_dim=model_config.get('embed_dim', 96),
            depths=model_config.get('depths', [2, 2, 6, 2])[:4],
            num_heads=model_config.get('num_heads', [3, 6, 12, 24])[:4],
            window_size=model_config.get('win_size', 8)
        )
        
        # Move to GPU and enable optimizations
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def compile_model(self):
        """Compile model with PyTorch 2.0 for A100 optimization"""
        if hasattr(torch, 'compile') and self.config.get('training', {}).get('compile_model', True):
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                self.logger.info("Model compiled with PyTorch 2.0 for A100 optimization")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
    
    def build_optimizer(self) -> optim.Optimizer:
        """Build optimizer"""
        opt_config = self.config['training']['optimizer']
        
        if opt_config['name'].lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config.get('weight_decay', 0.01),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['name']}")
        
        return optimizer
    
    def build_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler"""
        sched_config = self.config['training'].get('scheduler')
        if not sched_config:
            return None
            
        if sched_config['name'].lower() == 'cosineannealinglr':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_config['name'].lower() == 'reduceonplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10),
                min_lr=sched_config.get('min_lr', 1e-6)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_config['name']}")
            
        return scheduler
    
    def setup_experiment_tracking(self):
        """Setup tensorboard and checkpoints"""
        exp_name = self.config['experiment']['name']
        self.exp_dir = Path(self.config['experiment']['output_dir']) / exp_name
        
        # Create directories
        self.checkpoints_dir = self.exp_dir / 'checkpoints'
        self.logs_dir = self.exp_dir / 'logs' 
        self.tensorboard_dir = self.exp_dir / 'tensorboard'
        
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.tensorboard_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        
        # Save config
        with open(self.exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict()
        }
        
        # Regular checkpoint (formato completo per resume training)
        checkpoint_path = self.checkpoints_dir / f'checkpoint_epoch_{epoch:04d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Weights-only checkpoint (per inference/testing)
        weights_only_path = self.checkpoints_dir / f'weights_epoch_{epoch:04d}.pth'
        torch.save(self.model.state_dict(), weights_only_path)
        
        # Best model checkpoint (entrambi i formati)
        if is_best:
            # Formato completo
            best_path = self.checkpoints_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            
            # Weights-only
            best_weights_path = self.checkpoints_dir / 'best_weights.pth'
            torch.save(self.model.state_dict(), best_weights_path)
            
            self.logger.info(f"New best model saved at epoch {epoch} (both formats)")
        
        self.logger.info(f"Saved checkpoint epoch {epoch} (complete + weights-only)")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_losses = {'total': 0, 'image': 0, 'mask': 0}
        num_batches = len(self.train_loader)
        
        start_time = time.time()
        
        # Progress bar for training - clean formatting
        pbar = tqdm(enumerate(self.train_loader), total=num_batches, 
                   desc=f"Epoch {epoch}", leave=False, ncols=100, position=0, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        
        for batch_idx, batch in pbar:
            # Move to GPU
            input_imgs = batch['input'].to(self.device, non_blocking=True)
            target_imgs = batch['target'].to(self.device, non_blocking=True) 
            target_masks = batch['mask'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast():
                pred_starless, pred_mask = self.model(input_imgs)
                losses = self.criterion(pred_starless, pred_mask, target_imgs, target_masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(losses['total']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Accumulate losses
            for key, value in losses.items():
                total_losses[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{losses['total'].item():.4f}",
                'Img': f"{losses['image'].item():.4f}",
                'Mask': f"{losses['mask'].item():.4f}",
                'Speed': f"{(batch_idx + 1) * self.config['training']['batch_size'] / (time.time() - start_time):.1f} samples/s"
            })
            
            # Less frequent and cleaner logging 
            if batch_idx % (self.config['training'].get('log_interval', 50) * 10) == 0 and batch_idx > 0:
                batch_time = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * self.config['training']['batch_size'] / batch_time
                
                # Use pbar.write for clean logging without interfering with progress bar
                pbar.write(f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                          f"Loss: {losses['total'].item():.6f} "
                          f"Img: {losses['image'].item():.6f} "
                          f"Mask: {losses['mask'].item():.6f} "
                          f"Speed: {samples_per_sec:.1f} samples/s")
                
                # Tensorboard logging
                self.writer.add_scalar('Train/BatchLoss', losses['total'].item(), self.global_step)
                self.writer.add_scalar('Train/Speed', samples_per_sec, self.global_step)
            
            self.global_step += 1
        
        pbar.close()
        
        # Average losses
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        epoch_time = time.time() - start_time
        self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        return avg_losses
    
    def validate_epoch(self, epoch: int) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_losses = {'total': 0, 'image': 0, 'mask': 0}
        all_metrics = []
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            # Progress bar for validation
            pbar = tqdm(self.val_loader, desc="Validation", leave=False)
            for batch in pbar:
                # Move to GPU
                input_imgs = batch['input'].to(self.device, non_blocking=True)
                target_imgs = batch['target'].to(self.device, non_blocking=True)
                target_masks = batch['mask'].to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast():
                    pred_starless, pred_mask = self.model(input_imgs)
                    losses = self.criterion(pred_starless, pred_mask, target_imgs, target_masks)
                
                # Accumulate losses
                for key, value in losses.items():
                    total_losses[key] += value.item()
                
                # Compute metrics - clamp to [0,1] for proper evaluation
                pred_clamped = pred_starless.clamp(0, 1)
                target_clamped = target_imgs.clamp(0, 1)
                metrics = UFormerMetrics.compute_all_metrics(
                    pred_clamped, pred_mask, target_clamped, target_masks
                )
                all_metrics.append(metrics)
                
                # Update progress bar
                pbar.set_postfix({'Val Loss': f"{losses['total'].item():.4f}"})
            
            pbar.close()
        
        # Average losses and metrics
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        
        # Average metrics across all batches
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        return avg_losses, avg_metrics
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.current_epoch + 1, self.config['training']['epochs'] + 1):
            # Training
            train_losses = self.train_epoch(epoch)
            
            # Validation
            val_losses, val_metrics = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_losses['total'])
                else:
                    self.scheduler.step()
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.logger.info(
                f"Epoch {epoch}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_losses['total']:.6f} - "
                f"Val Loss: {val_losses['total']:.6f} - "
                f"Val PSNR: {val_metrics['psnr']:.2f} - "
                f"Val SSIM: {val_metrics['ssim']:.4f} - "
                f"LR: {current_lr:.2e}"
            )
            
            # Tensorboard logging
            for key, value in train_losses.items():
                self.writer.add_scalar(f'Train/{key.title()}Loss', value, epoch)
            for key, value in val_losses.items():
                self.writer.add_scalar(f'Val/{key.title()}Loss', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key.title()}', value, epoch)
            self.writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']
            
            # Save every epoch (not just at intervals)
            self.save_checkpoint(epoch, is_best)
            
            # Also save regular intervals
            if epoch % self.config['training'].get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        self.logger.info("Training completed!")
        self.writer.close()


def main():
    """Main function to start training"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_uformer.yaml',
                      help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = A100Trainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

"""
Fine-tuning Enhanced UFormer da checkpoint esistente
Training in 2 fasi per preservare conoscenza esistente

Fase 1: Solo nuovi moduli (Halo, Focal) - 2-3 epoche
Fase 2: Fine-tuning completo - 5-10 epoche

Per risolvere il problema dei quadretti mantenendo le performance esistenti.
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
from enhanced_loss import EnhancedUFormerLoss  # Nuova loss con perceptual

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedUFormerTrainer:
    """
    Trainer per Enhanced UFormer con fine-tuning in 2 fasi
    """
    
    def __init__(self, config_path: str, pretrained_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrained_path = pretrained_path
        
        # Setup modello Enhanced
        self.model = EnhancedUFormerStarRemoval(
            embed_dim=self.config['model']['embed_dim'],
            window_size=self.config['model']['win_size'],  # Ora leggibile da config
            halo_size=self.config['model'].get('halo_size', 4),  # Da config con default
            depths=self.config['model']['depths'],
            num_heads=self.config['model']['num_heads'],
            focal_interval=self.config['model'].get('focal_interval', 2),
            shifted_window=self.config['model'].get('shifted_window', True)  # NUOVO: abilita shifted windows
        ).to(self.device)
        
        # Carica checkpoint esistente (compatibile)
        self.load_pretrained()
        
        # Setup Enhanced loss con Perceptual Loss
        self.criterion = EnhancedUFormerLoss(
            l1_weight=1.0,                # Base L1 loss
            perceptual_weight=0.1,        # Perceptual VGG loss per qualità visiva
            ssim_weight=0.1,              # SSIM per similarità strutturale  
            mask_weight=self.config.get('mask_weight', 0.1),
            use_ssim=True                 # Abilita SSIM loss
        )
        
        # Setup datasets
        self.setup_datasets()
        
        # Setup experiment directory
        self.setup_experiment_dir()
        
    def load_pretrained(self):
        """Carica checkpoint esistente in modo compatibile"""
        logger.info(f"Loading pretrained model from {self.pretrained_path}")
        
        missing_keys, unexpected_keys = self.model.load_pretrained_compatible(
            self.pretrained_path, strict=False
        )
        
        logger.info(f"Successfully loaded pretrained model")
        logger.info(f"New modules (will be trained from scratch): {len(missing_keys)}")
        
        # Log alcuni missing keys per debug
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
        
        # Subset validation per speed
        val_subset_size = self.config['training'].get('val_subset_size', 1000)
        if len(self.val_dataset) > val_subset_size:
            indices = np.random.choice(len(self.val_dataset), val_subset_size, replace=False)
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, indices)
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['val_batch_size'],
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        
    def setup_experiment_dir(self):
        """Setup directory per salvare risultati"""
        exp_name = f"enhanced_uformer_finetune_{int(time.time())}"
        self.exp_dir = Path(self.config['experiment']['output_dir']) / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info(f"Experiment directory: {self.exp_dir}")
        
    def freeze_pretrained_modules(self):
        """Congela moduli pre-trainati per Fase 1"""
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.model.named_parameters():
            # Congela tutto eccetto moduli con "halo" o "focal" nel nome
            if 'halo' in name.lower() or 'focal' in name.lower():
                param.requires_grad = True
                trainable_count += param.numel()
            else:
                param.requires_grad = False
                frozen_count += param.numel()
        
        logger.info(f"Phase 1 - Frozen parameters: {frozen_count:,}")
        logger.info(f"Phase 1 - Trainable parameters: {trainable_count:,}")
        
    def unfreeze_all_modules(self):
        """Scongela tutti i moduli per Fase 2"""
        trainable_count = 0
        
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            trainable_count += param.numel()
        
        logger.info(f"Phase 2 - All trainable parameters: {trainable_count:,}")
    
    def train_epoch(self, optimizer, epoch, phase_name="Training"):
        """Training per una epoca"""
        self.model.train()
        total_loss = 0.0
        total_img_loss = 0.0
        total_mask_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"{phase_name} Epoch {epoch}")
        
        for batch_idx, (inputs, targets, masks) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device) 
            masks = masks.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred_starless, pred_mask = self.model(inputs)
            
            # Compute loss
            loss, loss_dict = self.criterion(pred_starless, pred_mask, targets, masks)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping per stabilità
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_img_loss += loss_dict.get('l1_loss', 0)
            total_mask_loss += loss_dict.get('mask_loss', 0)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Img': f'{total_img_loss/(batch_idx+1):.4f}',
                    'Mask': f'{total_mask_loss/(batch_idx+1):.4f}'
                })
        
        return total_loss / len(self.train_loader)\n    \n    def validate(self, epoch):\n        \"\"\"Validazione\"\"\"\n        self.model.eval()\n        total_loss = 0.0\n        \n        with torch.no_grad():\n            for inputs, targets, masks in tqdm(self.val_loader, desc=f\"Validation Epoch {epoch}\"):\n                inputs = inputs.to(self.device)\n                targets = targets.to(self.device)\n                masks = masks.to(self.device)\n                \n                pred_starless, pred_mask = self.model(inputs)\n                loss, _, _ = self.criterion(pred_starless, pred_mask, targets, masks)\n                \n                total_loss += loss.item()\n        \n        avg_loss = total_loss / len(self.val_loader)\n        return avg_loss\n    \n    def save_checkpoint(self, epoch, optimizer, loss, phase, is_best=False):\n        \"\"\"Salva checkpoint\"\"\"\n        checkpoint = {\n            'epoch': epoch,\n            'model_state_dict': self.model.state_dict(),\n            'optimizer_state_dict': optimizer.state_dict(),\n            'loss': loss,\n            'phase': phase,\n            'config': self.config\n        }\n        \n        # Checkpoint corrente\n        checkpoint_path = self.checkpoint_dir / f'enhanced_uformer_{phase}_epoch_{epoch:03d}.pth'\n        torch.save(checkpoint, checkpoint_path)\n        \n        # Best model\n        if is_best:\n            best_path = self.checkpoint_dir / f'enhanced_best_model_{phase}.pth'\n            torch.save(checkpoint, best_path)\n            logger.info(f\"New best model saved: {best_path}\")\n        \n        logger.info(f\"Checkpoint saved: {checkpoint_path}\")\n    \n    def train_phase_1(self):\n        \"\"\"Fase 1: Solo nuovi moduli Halo/Focal\"\"\"\n        logger.info(\"=== PHASE 1: Training only new Halo/Focal modules ===\")\n        \n        # Congela moduli pre-trainati\n        self.freeze_pretrained_modules()\n        \n        # Optimizer solo per moduli trainabili\n        trainable_params = [p for p in self.model.parameters() if p.requires_grad]\n        optimizer = optim.AdamW(\n            trainable_params,\n            lr=self.config['training']['optimizer']['lr'] * 0.1,  # LR ridotto\n            weight_decay=self.config['training']['optimizer']['weight_decay']\n        )\n        \n        scheduler = optim.lr_scheduler.CosineAnnealingLR(\n            optimizer, T_max=3, eta_min=1e-6\n        )\n        \n        best_val_loss = float('inf')\n        \n        for epoch in range(1, 4):  # 3 epoche per Fase 1\n            # Training\n            train_loss = self.train_epoch(optimizer, epoch, \"Phase1\")\n            \n            # Validation\n            val_loss = self.validate(epoch)\n            \n            # Scheduler step\n            scheduler.step()\n            \n            # Logging\n            logger.info(f\"Phase 1 Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\")\n            \n            # Save checkpoint\n            is_best = val_loss < best_val_loss\n            if is_best:\n                best_val_loss = val_loss\n            \n            self.save_checkpoint(epoch, optimizer, val_loss, \"phase1\", is_best)\n        \n        logger.info(f\"Phase 1 completed. Best val loss: {best_val_loss:.4f}\")\n        \n    def train_phase_2(self):\n        \"\"\"Fase 2: Fine-tuning completo\"\"\"\n        logger.info(\"=== PHASE 2: Full model fine-tuning ===\")\n        \n        # Scongela tutti i moduli\n        self.unfreeze_all_modules()\n        \n        # Optimizer per tutti i parametri con LR più basso\n        optimizer = optim.AdamW(\n            self.model.parameters(),\n            lr=self.config['training']['optimizer']['lr'] * 0.01,  # LR molto ridotto\n            weight_decay=self.config['training']['optimizer']['weight_decay']\n        )\n        \n        scheduler = optim.lr_scheduler.CosineAnnealingLR(\n            optimizer, T_max=10, eta_min=1e-7\n        )\n        \n        best_val_loss = float('inf')\n        \n        for epoch in range(1, 11):  # 10 epoche per Fase 2\n            # Training\n            train_loss = self.train_epoch(optimizer, epoch, \"Phase2\")\n            \n            # Validation\n            val_loss = self.validate(epoch)\n            \n            # Scheduler step\n            scheduler.step()\n            \n            # Logging\n            logger.info(f\"Phase 2 Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\")\n            \n            # Save checkpoint\n            is_best = val_loss < best_val_loss\n            if is_best:\n                best_val_loss = val_loss\n            \n            self.save_checkpoint(epoch, optimizer, val_loss, \"phase2\", is_best)\n        \n        logger.info(f\"Phase 2 completed. Best val loss: {best_val_loss:.4f}\")\n        \n    def run_full_training(self):\n        \"\"\"Esegue training completo in 2 fasi\"\"\"\n        logger.info(\"Starting Enhanced UFormer fine-tuning\")\n        start_time = time.time()\n        \n        try:\n            # Fase 1: Solo nuovi moduli\n            self.train_phase_1()\n            \n            # Fase 2: Fine-tuning completo\n            self.train_phase_2()\n            \n            # Summary\n            total_time = time.time() - start_time\n            logger.info(f\"Training completed in {total_time/3600:.2f} hours\")\n            logger.info(f\"Best models saved in: {self.checkpoint_dir}\")\n            \n        except KeyboardInterrupt:\n            logger.info(\"Training interrupted by user\")\n        except Exception as e:\n            logger.error(f\"Training failed: {str(e)}\")\n            raise\n\n\ndef main():\n    parser = argparse.ArgumentParser(description='Enhanced UFormer Fine-tuning')\n    parser.add_argument('--config', type=str, required=True, help='Config file path')\n    parser.add_argument('--pretrained', type=str, required=True, help='Pretrained model path')\n    \n    args = parser.parse_args()\n    \n    # Crea trainer e avvia training\n    trainer = EnhancedUFormerTrainer(args.config, args.pretrained)\n    trainer.run_full_training()\n\n\nif __name__ == '__main__':\n    main()"

"""
Loss functions and metrics for UFormer dual-head star removal
Designed for simplicity and effectiveness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


class UFormerLoss(nn.Module):
    """
    Combined loss for dual-head UFormer:
    - L1 loss for starless image reconstruction  
    - BCE loss for star mask prediction
    """
    
    def __init__(self, mask_weight: float = 0.1):
        super().__init__()
        self.mask_weight = mask_weight
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, pred_starless: torch.Tensor, pred_mask: torch.Tensor, 
                target_starless: torch.Tensor, target_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_starless: [B, 3, H, W] predicted starless image
            pred_mask: [B, 1, H, W] predicted star mask (logits)
            target_starless: [B, 3, H, W] target starless image
            target_mask: [B, 1, H, W] target star mask (0-1)
        """
        
        # Image reconstruction loss
        img_loss = self.l1_loss(pred_starless, target_starless)
        
        # Star mask prediction loss
        mask_loss = self.bce_loss(pred_mask, target_mask)
        
        # Total loss
        total_loss = img_loss + self.mask_weight * mask_loss
        
        return {
            'total': total_loss,
            'image': img_loss,
            'mask': mask_loss
        }


class UFormerMetrics:
    """
    Metrics for evaluating UFormer performance
    """
    
    @staticmethod
    def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute PSNR between predictions and targets"""
        # Convert to numpy and handle batch dimension
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        psnr_values = []
        for i in range(pred_np.shape[0]):
            # Convert from [C, H, W] to [H, W, C] for skimage
            pred_img = np.transpose(pred_np[i], (1, 2, 0))
            target_img = np.transpose(target_np[i], (1, 2, 0))
            
            # Clip to valid range
            pred_img = np.clip(pred_img, 0, 1)
            target_img = np.clip(target_img, 0, 1)
            
            psnr_val = psnr(target_img, pred_img, data_range=1.0)
            psnr_values.append(psnr_val)
            
        return float(np.mean(psnr_values))
    
    @staticmethod  
    def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute SSIM between predictions and targets"""
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        ssim_values = []
        for i in range(pred_np.shape[0]):
            # Convert from [C, H, W] to [H, W, C] for skimage
            pred_img = np.transpose(pred_np[i], (1, 2, 0))
            target_img = np.transpose(target_np[i], (1, 2, 0))
            
            # Clip to valid range
            pred_img = np.clip(pred_img, 0, 1)
            target_img = np.clip(target_img, 0, 1)
            
            ssim_val = ssim(target_img, pred_img, data_range=1.0, 
                           channel_axis=-1)
            ssim_values.append(ssim_val)
            
        return float(np.mean(ssim_values))
    
    @staticmethod
    def compute_mask_metrics(pred_mask: torch.Tensor, target_mask: torch.Tensor, 
                           threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute precision, recall, F1 for star mask prediction
        
        Args:
            pred_mask: [B, 1, H, W] predicted mask logits
            target_mask: [B, 1, H, W] target mask (0-1)
        """
        # Convert logits to probabilities and threshold
        pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
        target_binary = target_mask
        
        # Flatten for easier computation
        pred_flat = pred_binary.view(-1)
        target_flat = target_binary.view(-1)
        
        # Compute metrics
        tp = (pred_flat * target_flat).sum().item()
        fp = (pred_flat * (1 - target_flat)).sum().item()
        fn = ((1 - pred_flat) * target_flat).sum().item()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def compute_all_metrics(pred_starless: torch.Tensor, pred_mask: torch.Tensor,
                          target_starless: torch.Tensor, target_mask: torch.Tensor) -> Dict[str, float]:
        """Compute all metrics for UFormer evaluation"""
        
        metrics = {}
        
        # Image quality metrics
        metrics['psnr'] = UFormerMetrics.compute_psnr(pred_starless, target_starless)
        metrics['ssim'] = UFormerMetrics.compute_ssim(pred_starless, target_starless)
        
        # Mask prediction metrics
        mask_metrics = UFormerMetrics.compute_mask_metrics(pred_mask, target_mask)
        metrics.update({f'mask_{k}': v for k, v in mask_metrics.items()})
        
        return metrics


def test_loss_and_metrics():
    """Test the loss and metrics implementations"""
    
    # Create dummy data
    batch_size = 4
    channels = 3
    height, width = 512, 512
    
    pred_starless = torch.randn(batch_size, channels, height, width)
    pred_mask = torch.randn(batch_size, 1, height, width)  # logits
    target_starless = torch.randn(batch_size, channels, height, width)
    target_mask = torch.randint(0, 2, (batch_size, 1, height, width)).float()
    
    # Test loss
    criterion = UFormerLoss(mask_weight=0.1)
    losses = criterion(pred_starless, pred_mask, target_starless, target_mask)
    
    print("Loss computation:")
    for name, value in losses.items():
        print(f"  {name}: {value.item():.6f}")
    
    # Test metrics
    # Normalize predictions to [0, 1] for metrics
    pred_starless_norm = torch.sigmoid(pred_starless)
    target_starless_norm = torch.sigmoid(target_starless)
    
    metrics = UFormerMetrics.compute_all_metrics(
        pred_starless_norm, pred_mask, target_starless_norm, target_mask
    )
    
    print("\nMetrics computation:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    
    print("\nLoss and metrics test completed successfully!")


if __name__ == "__main__":
    test_loss_and_metrics()

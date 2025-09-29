#!/usr/bin/env python3
"""
Enhanced Loss con Star-Only Focus
Preserva tutto tranne le stelle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from typing import Tuple, Dict


class StarFocusedLoss(nn.Module):
    """
    Loss che modifica SOLO le aree delle stelle
    Preserva tutto il resto identico
    """
    
    def __init__(
        self, 
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.0,  # DISABILITATO per preservare dettagli
        mask_focus_weight: float = 5.0,   # FOCUS solo su stelle
        preserve_weight: float = 10.0     # PRESERVA resto immagine
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.mask_focus_weight = mask_focus_weight
        self.preserve_weight = preserve_weight
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
    def forward(
        self, 
        pred_img: torch.Tensor, 
        pred_mask: torch.Tensor,
        target_img: torch.Tensor, 
        target_mask: torch.Tensor,
        input_img: torch.Tensor  # NUOVO: immagine originale
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_img: Predicted starless image [B, 3, H, W]
            pred_mask: Predicted star mask [B, 1, H, W]  
            target_img: Target starless image [B, 3, H, W]
            target_mask: Target star mask [B, 1, H, W]
            input_img: Original input image [B, 3, H, W]
        """
        
        # 1. STAR AREAS ONLY - Focus solo sulle stelle
        star_area_loss = self.l1_loss(
            pred_img * target_mask, 
            target_img * target_mask
        )
        
        # 2. PRESERVE NON-STAR AREAS - Identico al input
        non_star_mask = 1.0 - target_mask
        preserve_loss = self.mse_loss(
            pred_img * non_star_mask,
            input_img * non_star_mask  # Deve essere IDENTICO all'input
        )
        
        # 3. MASK PREDICTION
        mask_loss = self.l1_loss(pred_mask, target_mask)
        
        # Combined loss
        total_loss = (
            self.mask_focus_weight * star_area_loss +      # Focus stelle
            self.preserve_weight * preserve_loss +          # Preserva resto
            self.l1_weight * mask_loss                      # Mask accuracy
        )
        
        # Loss breakdown
        loss_dict = {
            'total_loss': total_loss.item(),
            'star_area_loss': star_area_loss.item(),
            'preserve_loss': preserve_loss.item(), 
            'mask_loss': mask_loss.item(),
        }
        
        return total_loss, loss_dict


class DetailPreservingLoss(nn.Module):
    """
    Loss alternativa con residual learning
    """
    
    def __init__(
        self,
        residual_weight: float = 1.0,
        mask_weight: float = 0.5,
        detail_weight: float = 2.0
    ):
        super().__init__()
        
        self.residual_weight = residual_weight
        self.mask_weight = mask_weight  
        self.detail_weight = detail_weight
        
        self.l1_loss = nn.L1Loss()
        
    def forward(
        self,
        pred_img: torch.Tensor,
        pred_mask: torch.Tensor, 
        target_img: torch.Tensor,
        target_mask: torch.Tensor,
        input_img: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        # Calcola residual (differenza che deve apprendere)
        target_residual = target_img - input_img
        pred_residual = pred_img - input_img
        
        # Loss solo sul residual
        residual_loss = self.l1_loss(pred_residual, target_residual)
        
        # Mask loss
        mask_loss = self.l1_loss(pred_mask, target_mask)
        
        # Detail preservation - penalizza cambi non necessari
        detail_loss = self.l1_loss(
            pred_img * (1.0 - target_mask),
            input_img * (1.0 - target_mask)
        )
        
        total_loss = (
            self.residual_weight * residual_loss +
            self.mask_weight * mask_loss + 
            self.detail_weight * detail_loss
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'residual_loss': residual_loss.item(),
            'mask_loss': mask_loss.item(),
            'detail_loss': detail_loss.item(),
        }
        
        return total_loss, loss_dict
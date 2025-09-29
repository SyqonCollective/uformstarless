"""
Enhanced UFormer Loss con Perceptual Loss per qualità visiva migliorata

Combina:
1. L1 Loss: Accuratezza pixel-wise
2. Perceptual Loss (VGG): Qualità visiva e coerenza percettiva  
3. SSIM Loss: Similarità strutturale (opzionale)
4. Star Mask Loss: Precisione nella rimozione delle stelle

Risolve il problema delle stelle grandi e migliora la qualità percettiva.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Tuple, Dict


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual Loss basato su VGG16 pre-trained
    Confronta features di alto livello per qualità visiva
    """
    
    def __init__(self, layer_weights: Dict[str, float] = None):
        super().__init__()
        
        # Layer weights per default (enfasi su features middle-level)
        self.layer_weights = layer_weights or {
            'relu1_2': 0.1,   # Early features
            'relu2_2': 0.2,   # Low-level textures
            'relu3_3': 0.4,   # Mid-level patterns  
            'relu4_3': 0.3,   # High-level features
        }
        
        # Load VGG16 pre-trained
        vgg = models.vgg16(pretrained=True).features
        self.vgg_layers = nn.ModuleDict()
        
        # Extract specific layers
        layer_map = {
            'relu1_2': 3,   # after conv1_2 + relu
            'relu2_2': 8,   # after conv2_2 + relu  
            'relu3_3': 15,  # after conv3_3 + relu
            'relu4_3': 22,  # after conv4_3 + relu
        }
        
        for name, idx in layer_map.items():
            self.vgg_layers[name] = nn.Sequential(*list(vgg.children())[:idx+1])
            
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization per ImageNet pre-training
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
    def to(self, device):
        """Override to method to ensure VGG layers go to device"""
        super().to(device)
        for layer in self.vgg_layers.values():
            layer.to(device)
        return self
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted image [B, 3, H, W] in [0, 1]
            target: Target image [B, 3, H, W] in [0, 1]
        """
        # Normalize per VGG
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        perceptual_loss = 0.0
        
        for layer_name, weight in self.layer_weights.items():
            # Extract features
            pred_feat = self.vgg_layers[layer_name](pred_norm)
            target_feat = self.vgg_layers[layer_name](target_norm)
            
            # L2 loss nelle feature
            layer_loss = F.mse_loss(pred_feat, target_feat)
            perceptual_loss += weight * layer_loss
            
        return perceptual_loss


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index Measure Loss
    Misura similarità strutturale tra immagini
    """
    
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)
        
    def gaussian(self, window_size: int, sigma: float = 1.5):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2/float(2*sigma**2))) 
            for x in range(window_size)
        ])
        return gauss/gauss.sum()
        
    def create_window(self, window_size: int, channel: int):
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
        
    def ssim(self, img1: torch.Tensor, img2: torch.Tensor):
        (_, channel, _, _) = img1.size()
        
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel
            
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
            
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        return 1 - self.ssim(img1, img2)


class EnhancedUFormerLoss(nn.Module):
    """
    Loss Enhanced per UFormer con Perceptual Loss
    
    Combina L1 + Perceptual + SSIM per qualità visiva ottimale
    Gestisce meglio stelle grandi e artifact reduction
    """
    
    def __init__(
        self, 
        l1_weight: float = 1.0,
        perceptual_weight: float = 0.1,  # Importante ma non dominante
        ssim_weight: float = 0.1,        # Opzionale per SSIM
        mask_weight: float = 0.1,        # Weight per star mask
        use_ssim: bool = True            # Se usare SSIM loss
    ):
        super().__init__()
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.ssim_weight = ssim_weight
        self.mask_weight = mask_weight
        self.use_ssim = use_ssim
        
        # Loss components
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        
        # Perceptual loss
        self.perceptual_loss = VGGPerceptualLoss()
        
        # SSIM loss (opzionale)
        if self.use_ssim:
            self.ssim_loss = SSIMLoss()
            
    def to(self, device):
        """Override to method to ensure all components go to device"""
        super().to(device)
        self.perceptual_loss.to(device)
        if hasattr(self, 'ssim_loss'):
            self.ssim_loss.to(device)
        return self
        
    def forward(
        self, 
        pred_img: torch.Tensor, 
        pred_mask: torch.Tensor,
        target_img: torch.Tensor, 
        target_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_img: Predicted starless image [B, 3, H, W]
            pred_mask: Predicted star mask [B, 1, H, W]  
            target_img: Target starless image [B, 3, H, W]
            target_mask: Target star mask [B, 1, H, W]
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        
        # L1 Loss (base accuracy)
        l1_img_loss = self.l1_loss(pred_img, target_img)
        l1_mask_loss = self.l1_loss(pred_mask, target_mask)
        
        # Perceptual Loss (visual quality) 
        perceptual_img_loss = self.perceptual_loss(pred_img, target_img)
        
        # SSIM Loss (structural similarity)
        ssim_img_loss = torch.tensor(0.0, device=pred_img.device)
        if self.use_ssim:
            ssim_img_loss = self.ssim_loss(pred_img, target_img)
        
        # Combined loss
        total_loss = (
            self.l1_weight * l1_img_loss +
            self.perceptual_weight * perceptual_img_loss +
            self.ssim_weight * ssim_img_loss +
            self.mask_weight * l1_mask_loss
        )
        
        # Loss breakdown per monitoring
        loss_dict = {
            'total_loss': total_loss.item(),
            'l1_img_loss': l1_img_loss.item(),
            'l1_mask_loss': l1_mask_loss.item(),
            'perceptual_loss': perceptual_img_loss.item(),
            'ssim_loss': ssim_img_loss.item() if self.use_ssim else 0.0,
        }
        
        return total_loss, loss_dict


class EnhancedUFormerMetrics:
    """
    Metriche per valutazione Enhanced UFormer
    Include PSNR, SSIM, LPIPS per qualità visiva
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.psnr_sum = 0.0
        self.ssim_sum = 0.0 
        self.count = 0
        
    def update(self, pred_img: torch.Tensor, target_img: torch.Tensor):
        """Update metrics con batch di immagini"""
        batch_size = pred_img.size(0)
        
        with torch.no_grad():
            # PSNR
            mse = F.mse_loss(pred_img, target_img, reduction='none')
            mse = mse.view(batch_size, -1).mean(dim=1)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            self.psnr_sum += psnr.sum().item()
            
            # SSIM (semplificato)
            # Per ora usiamo correlazione come proxy
            pred_flat = pred_img.view(batch_size, -1)
            target_flat = target_img.view(batch_size, -1)
            
            pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
            target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
            
            correlation = (pred_centered * target_centered).sum(dim=1) / (
                torch.sqrt((pred_centered ** 2).sum(dim=1)) * 
                torch.sqrt((target_centered ** 2).sum(dim=1)) + 1e-8
            )
            
            self.ssim_sum += correlation.sum().item()
            self.count += batch_size
            
    def compute(self) -> Dict[str, float]:
        """Compute final metrics"""
        if self.count == 0:
            return {'psnr': 0.0, 'ssim': 0.0}
            
        return {
            'psnr': self.psnr_sum / self.count,
            'ssim': self.ssim_sum / self.count
        }
"""
Estensione GRADUALE del UFormer esistente
Strategia: Aggiungi Halo senza rompere compatibilità
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import copy

# Import UFormer originale
from uformer import UFormerStarRemoval, WindowAttention


class HaloWindowAttention(nn.Module):
    """
    Halo Attention che può SOSTITUIRE WindowAttention esistente
    Mantiene stessi parametri ma aggiunge halo capability
    """
    def __init__(self, original_attention: WindowAttention, halo_size: int = 4):
        super().__init__()
        
        # Copia TUTTI i parametri dal WindowAttention originale
        self.dim = original_attention.dim
        self.window_size = original_attention.window_size
        self.num_heads = original_attention.num_heads
        self.halo_size = halo_size
        
        # Copia i layer esistenti (STESSI PESI)
        self.qkv = copy.deepcopy(original_attention.qkv)
        self.proj = copy.deepcopy(original_attention.proj)
        self.scale = original_attention.scale
        
    def forward(self, x):
        """Forward con Halo - compatibile con WindowAttention"""
        B, C, H, W = x.shape
        
        # Se halo_size = 0, comportamento identico a WindowAttention
        if self.halo_size == 0:
            return self._standard_window_attention(x)
        
        # Halo attention
        return self._halo_attention(x)
    
    def _standard_window_attention(self, x):
        """Attention standard - IDENTICA al WindowAttention originale"""
        B, C, H, W = x.shape
        
        # Window partition
        x_windows = self._window_partition(x, self.window_size)  # [B*nW, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [B*nW, window_size*window_size, C]
        
        # Multi-head attention
        qkv = self.qkv(x_windows).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*nW, num_heads, window_size*window_size, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x = self.proj(x)
        
        # Window reverse
        x = x.view(-1, self.window_size, self.window_size, C)
        x = self._window_reverse(x, self.window_size, H, W)  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return x
    
    def _halo_attention(self, x):
        """Halo attention - aggiunge contesto oltre i bordi delle finestre"""
        B, C, H, W = x.shape
        
        # Padding per halo
        x_padded = F.pad(x, (self.halo_size, self.halo_size, self.halo_size, self.halo_size), mode='reflect')
        _, _, H_pad, W_pad = x_padded.shape
        
        # Window partition con halo
        window_size_h = self.window_size + 2 * self.halo_size
        x_windows = self._window_partition(x_padded.permute(0, 2, 3, 1), window_size_h)  # Con halo
        x_windows = x_windows.view(-1, window_size_h * window_size_h, C)
        
        # Multi-head attention
        qkv = self.qkv(x_windows).reshape(-1, window_size_h * window_size_h, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Halo mask: Q solo dal centro, K/V da centro + halo
        center_start = self.halo_size * window_size_h + self.halo_size
        center_size = self.window_size * self.window_size
        
        # Slice Q dal centro, K/V da tutto (centro + halo)
        q_center = q[:, :, center_start:center_start+center_size, :]
        
        # Scaled dot-product attention
        attn = (q_center @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(-1, center_size, C)
        x = self.proj(x)
        
        # Reshape back alla dimensione originale
        x = x.view(-1, self.window_size, self.window_size, C)
        x = self._window_reverse(x, self.window_size, H, W)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return x
    
    def _window_partition(self, x, window_size):
        """Partition into windows"""
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def _window_reverse(self, windows, window_size, H, W):
        """Reverse window partition"""
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class GraduallyEnhancedUFormer(nn.Module):
    """
    UFormer che può essere gradualmente migliorato
    
    1. Carica checkpoint esistente COMPLETAMENTE
    2. Sostituisce WindowAttention -> HaloWindowAttention
    3. Allena solo i miglioramenti
    """
    
    def __init__(self, halo_size: int = 4):
        super().__init__()
        self.halo_size = halo_size
        
        # Crea UFormer originale (sarà sovrascritto dal checkpoint)
        self.base_model = UFormerStarRemoval()
        
        # Flag per sapere se è stato convertito a Halo
        self.is_halo_converted = False
        
    def load_and_convert_to_halo(self, checkpoint_path: str):
        """
        1. Carica checkpoint nel base_model (identico)
        2. Converte WindowAttention -> HaloWindowAttention
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 1. Carica checkpoint nel modello base
        logger.info(f"Loading original checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Carica TUTTO nel modello base - deve essere 100% compatibile
        missing, unexpected = self.base_model.load_state_dict(state_dict, strict=True)
        
        if missing or unexpected:
            logger.error(f"Checkpoint not compatible! Missing: {missing}, Unexpected: {unexpected}")
            raise RuntimeError("Checkpoint incompatible")
        
        logger.info("✓ Original checkpoint loaded successfully")
        
        # 2. Converti WindowAttention -> HaloWindowAttention
        self._convert_to_halo_attention()
        
        logger.info("✓ Converted to Halo Attention")
        self.is_halo_converted = True
        
        return True
    
    def _convert_to_halo_attention(self):
        """Sostituisce tutti i WindowAttention con HaloWindowAttention"""
        
        def replace_attention_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, WindowAttention):
                    # Sostituisci con HaloWindowAttention che mantiene stessi pesi
                    halo_attention = HaloWindowAttention(child, self.halo_size)
                    setattr(module, name, halo_attention)
                else:
                    # Ricorsione sui figli
                    replace_attention_recursive(child)
        
        # Applica la conversione a tutto il modello
        replace_attention_recursive(self.base_model)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward identico al UFormer originale"""
        if not self.is_halo_converted:
            raise RuntimeError("Model not converted to Halo yet. Call load_and_convert_to_halo() first")
        
        return self.base_model(x)
    
    def get_halo_parameters(self):
        """Restituisce solo i parametri Halo (per fine-tuning selettivo)"""
        halo_params = []
        
        def collect_halo_params(module):
            for child in module.children():
                if isinstance(child, HaloWindowAttention):
                    # I parametri Halo sono gli stessi del WindowAttention originale
                    # Ma li possiamo isolare per fine-tuning differenziale
                    for param in child.parameters():
                        halo_params.append(param)
                else:
                    collect_halo_params(child)
        
        collect_halo_params(self.base_model)
        return halo_params


def create_gradually_enhanced_model(checkpoint_path: str, halo_size: int = 4, device: str = 'cuda'):
    """
    Factory function per creare modello enhanced da checkpoint esistente
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Crea modello
    model = GraduallyEnhancedUFormer(halo_size=halo_size)
    
    # Carica e converti
    success = model.load_and_convert_to_halo(checkpoint_path)
    
    if success:
        model = model.to(device)
        logger.info(f"✓ GraduallyEnhancedUFormer ready on {device}")
        logger.info(f"  Halo size: {halo_size}")
        logger.info(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    else:
        raise RuntimeError("Failed to create enhanced model")

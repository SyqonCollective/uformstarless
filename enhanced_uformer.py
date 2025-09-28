"""
Enhanced UFormer con Halo Attention e Cross-Window Focal Blocks
Compatibile con checkpoint esistenti tramite strict=False loading

Miglioramenti:
1. Halo Attention (window_size=16, halo=8) per ridurre quadretti
2. Shifted windows per comunicazione cross-window  
3. Focal blocks ogni 2 blocchi per stelle giganti
4. Compatibilità backward con modello originale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

# Import dei nuovi moduli
from halo_attention import HaloAttention, ShiftedHaloAttention
from cross_window_focal import FocalTransformerBlock


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class EnhancedUFormerBlock(nn.Module):
    """
    Blocco UFormer migliorato con Halo Attention
    """
    def __init__(
        self, 
        dim: int, 
        window_size: int = 16,  # Aumentato da 8 a 16 
        halo_size: int = 8,     # Halo per vedere oltre window
        num_heads: int = 8, 
        mlp_ratio: float = 4.0,
        shift: bool = False,    # Se usare shifted windows
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift
        
        self.norm1 = LayerNorm2d(dim)
        
        # Usa Halo Attention invece di Window Attention standard
        self.attn = ShiftedHaloAttention(
            dim=dim,
            window_size=window_size,
            halo_size=halo_size,
            num_heads=num_heads,
            shift=shift,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        
        self.norm2 = LayerNorm2d(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Halo Attention
        shortcut = x
        x = self.norm1(x)
        
        # Converti in formato (B, H, W, C) per attention
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.attn(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # Back to (B, C, H, W)
        
        x = shortcut + x
        
        # MLP
        shortcut = x
        x = self.norm2(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)  # [B*H*W, C]
        x = self.mlp(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        x = shortcut + x
        
        return x


class FocalUFormerBlock(nn.Module):
    """
    Blocco Focal che si inserisce ogni 2 blocchi normali
    Per gestire stelle molto grandi che superano qualsiasi window
    """
    def __init__(
        self,
        dim: int,
        window_size: int = 16,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        focal_window: int = 1,  # Raggio finestre vicine
        focal_level: int = 2,   # Livelli di focus
        drop: float = 0.0,
        attn_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        
        self.norm1 = LayerNorm2d(dim)
        
        self.focal_block = FocalTransformerBlock(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            focal_window=focal_window,
            focal_level=focal_level,
            drop=drop,
            attn_drop=attn_drop
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] 
        B, C, H, W = x.shape
        
        shortcut = x
        x = self.norm1(x)
        
        # Convert to (B, H, W, C) for focal attention
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.focal_block(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # Back to (B, C, H, W)
        
        x = shortcut + x
        
        return x


class EnhancedUFormerStarRemoval(nn.Module):
    """
    UFormer Enhanced per rimozione stelle senza quadretti
    
    Architettura:
    - Encoder-Decoder con skip connections
    - Halo Attention blocks con shifted windows
    - Focal blocks ogni 2 blocchi per stelle giganti
    - Dual head: starless image + star mask
    
    Compatibilità:
    - Carica checkpoint esistenti con strict=False
    - Solo i nuovi moduli partono da zero
    """
    
    def __init__(
        self, 
        embed_dim: int = 96,
        window_size: int = 8,  # Tornato a 8 per inferenza leggera
        halo_size: int = 4,    # Ridotto proporzionalmente 
        depths: list = [2, 2, 6, 2],  # Profondità aumentata nel livello centrale
        num_heads: list = [3, 6, 12, 24],
        focal_interval: int = 2,  # Ogni 2 blocchi, inserisci focal
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        patch_size: int = 4,
        shifted_window: bool = True  # NUOVO: abilita shifted windows per eliminare quadretti
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.focal_interval = focal_interval
        self.shifted_window = shifted_window  # Store for propagation to blocks
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer_depth = depths[i_layer]
            layer_heads = num_heads[i_layer]
            
            # Downsample (eccetto primo layer)
            if i_layer > 0:
                self.encoder_layers.append(
                    nn.Conv2d(int(embed_dim * 2 ** (i_layer-1)), layer_dim, 
                             kernel_size=2, stride=2)
                )
            
            # Enhanced blocks con mix di standard e focal
            layer_blocks = nn.ModuleList()
            for i_block in range(layer_depth):
                # Alterna tra shifted e non-shifted SOLO se shifted_window=True
                shift = (i_block % 2 == 1) and self.shifted_window
                
                # Ogni focal_interval blocchi, aggiungi focal block
                if (i_block + 1) % self.focal_interval == 0:
                    layer_blocks.append(
                        FocalUFormerBlock(
                            dim=layer_dim,
                            window_size=window_size,
                            num_heads=layer_heads,
                            mlp_ratio=mlp_ratio,
                            focal_window=1,  # 1 anello di finestre
                            focal_level=2,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate
                        )
                    )
                else:
                    # Blocco standard con Halo Attention
                    layer_blocks.append(
                        EnhancedUFormerBlock(
                            dim=layer_dim,
                            window_size=window_size,
                            halo_size=halo_size,
                            num_heads=layer_heads,
                            mlp_ratio=mlp_ratio,
                            shift=shift,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate
                        )
                    )
            
            self.encoder_layers.append(layer_blocks)
        
        # Bottleneck
        bottleneck_dim = int(embed_dim * 2 ** (self.num_layers - 1))
        self.bottleneck = EnhancedUFormerBlock(
            dim=bottleneck_dim,
            window_size=window_size,
            halo_size=halo_size,
            num_heads=num_heads[-1],
            mlp_ratio=mlp_ratio,
            shift=False,
            drop=drop_rate,
            attn_drop=attn_drop_rate
        )
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers - 1, -1, -1):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer_depth = depths[i_layer]
            layer_heads = num_heads[i_layer]
            
            # Upsample (eccetto ultimo layer)
            if i_layer < self.num_layers - 1:
                self.decoder_layers.append(
                    nn.ConvTranspose2d(int(embed_dim * 2 ** (i_layer+1)), layer_dim,
                                      kernel_size=2, stride=2)
                )
                
                # Skip connection fusion
                self.decoder_layers.append(
                    nn.Conv2d(layer_dim * 2, layer_dim, kernel_size=1)
                )
            
            # Decoder blocks
            layer_blocks = nn.ModuleList()
            for i_block in range(layer_depth):
                shift = (i_block % 2 == 1) and self.shifted_window
                
                if (i_block + 1) % self.focal_interval == 0:
                    layer_blocks.append(
                        FocalUFormerBlock(
                            dim=layer_dim,
                            window_size=window_size,
                            num_heads=layer_heads,
                            mlp_ratio=mlp_ratio,
                            focal_window=1,
                            focal_level=2,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate
                        )
                    )
                else:
                    layer_blocks.append(
                        EnhancedUFormerBlock(
                            dim=layer_dim,
                            window_size=window_size,
                            halo_size=halo_size,
                            num_heads=layer_heads,
                            mlp_ratio=mlp_ratio,
                            shift=shift,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate
                        )
                    )
            
            self.decoder_layers.append(layer_blocks)
        
        # Output heads
        self.patch_unembed = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Dual heads: starless image + star mask
        self.starless_head = nn.Conv2d(embed_dim, 3, kernel_size=3, padding=1)
        self.mask_head = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input: [B, 3, H, W]
        B, _, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/4, W/4]
        
        # Encoder con skip connections
        skip_connections = []
        
        layer_idx = 0
        for i_layer in range(self.num_layers):
            # Downsample se necessario
            if i_layer > 0:
                x = self.encoder_layers[layer_idx](x)
                layer_idx += 1
            
            # Prima di processare, salva per skip connection
            skip_connections.append(x)
            
            # Processa blocks
            blocks = self.encoder_layers[layer_idx]
            for block in blocks:
                x = block(x)
            layer_idx += 1
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder con skip connections
        layer_idx = 0
        for i_layer in range(self.num_layers - 1, -1, -1):
            # Upsample e skip connection se necessario
            if i_layer < self.num_layers - 1:
                x = self.decoder_layers[layer_idx](x)  # Upsample
                layer_idx += 1
                
                # Skip connection
                skip = skip_connections[i_layer]
                x = torch.cat([x, skip], dim=1)
                x = self.decoder_layers[layer_idx](x)  # Fuse
                layer_idx += 1
            
            # Decoder blocks
            blocks = self.decoder_layers[layer_idx]
            for block in blocks:
                x = block(x)
            layer_idx += 1
        
        # Patch unembed
        x = self.patch_unembed(x)  # [B, embed_dim, H, W]
        
        # Dual outputs
        starless = torch.sigmoid(self.starless_head(x))  # [B, 3, H, W]
        mask = torch.sigmoid(self.mask_head(x))          # [B, 1, H, W]
        
        return starless, mask
    
    def load_pretrained_compatible(self, checkpoint_path: str, strict: bool = False):
        """
        Carica checkpoint esistente in modo compatibile
        I nuovi moduli (Halo, Focal) partiranno da inizializzazione casuale
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Carica con strict=False per ignorare moduli mancanti/extra
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
        
        print(f"Loaded pretrained model from {checkpoint_path}")
        print(f"Missing keys (new modules): {len(missing_keys)}")
        print(f"Unexpected keys (removed modules): {len(unexpected_keys)}")
        
        return missing_keys, unexpected_keys

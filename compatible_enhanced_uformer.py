"""
Enhanced UFormer COMPATIBILE con checkpoint esistente
Aggiunge solo Halo Attention senza cambiare dimensioni dei layer

Mantiene:
- Stessa architettura encoder-decoder
- Stesse dimensioni dei layer 
- Sostituisce solo WindowAttention -> HaloAttention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Import UFormer originale
from uformer import LayerNorm2d, MLP


class HaloWindowAttention(nn.Module):
    """
    Halo Attention compatibile che sostituisce WindowAttention
    Mantiene stessa interfaccia ma aggiunge halo per vedere oltre finestre
    """
    def __init__(self, dim, window_size=8, halo_size=4, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Stessi layer del WindowAttention originale
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x: [B, C, H, W] - stesso input del WindowAttention originale
        B, C, H, W = x.shape
        
        # Padding per halo
        if self.halo_size > 0:
            x_padded = F.pad(x, (self.halo_size, self.halo_size, self.halo_size, self.halo_size), mode='reflect')
        else:
            x_padded = x
            
        # Converti in formato windows con halo
        x = x_padded.permute(0, 2, 3, 1)  # [B, H+2*halo, W+2*halo, C]
        H_pad, W_pad = x.shape[1], x.shape[2]
        
        # Window partitioning con halo
        window_size_with_halo = self.window_size + 2 * self.halo_size
        nH, nW = H // self.window_size, W // self.window_size
        
        # Reshape in windows
        x = x.view(B, nH, window_size_with_halo, nW, window_size_with_halo, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, window_size_with_halo * window_size_with_halo, C)
        
        # Self-attention (stesso del WindowAttention originale)
        qkv = self.qkv(x).reshape(-1, window_size_with_halo * window_size_with_halo, 
                                  3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Mask per dare più peso al centro (query) vs halo
        if self.halo_size > 0:
            center_start = self.halo_size * window_size_with_halo + self.halo_size
            center_end = center_start + self.window_size * self.window_size
            
            # Crea mask che penalizza attention dal centro verso l'halo
            mask = torch.zeros_like(attn)
            mask[:, :, center_start:center_end, :center_start] = -0.1  # Penalty verso halo sopra
            mask[:, :, center_start:center_end, center_end:] = -0.1    # Penalty verso halo sotto
            attn = attn + mask
        
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(-1, window_size_with_halo * window_size_with_halo, C)
        x = self.proj(x)
        
        # Estrai solo la parte centrale (rimuovi halo)
        if self.halo_size > 0:
            x = x.view(-1, window_size_with_halo, window_size_with_halo, C)
            start_idx = self.halo_size
            end_idx = start_idx + self.window_size
            x = x[:, start_idx:end_idx, start_idx:end_idx, :]
            x = x.reshape(-1, self.window_size * self.window_size, C)
        
        # Reshape back (stesso del WindowAttention originale)
        x = x.view(B, nH, nW, self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return x


class EnhancedUFormerBlock(nn.Module):
    """
    UFormer Block con Halo Attention invece di Window Attention
    COMPATIBILE con checkpoint esistente - stesse dimensioni
    """
    def __init__(self, dim, window_size=8, num_heads=8, mlp_ratio=4.0, 
                 halo_size=4, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        
        # Stessi layer del UFormerBlock originale
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        
        # SOSTITUISCE WindowAttention con HaloWindowAttention  
        self.attn = HaloWindowAttention(dim, window_size, halo_size, num_heads)
        
    def forward(self, x):
        # x: [B, C, H, W] - stesso input del UFormerBlock originale
        B, C, H, W = x.shape
        
        # Shifted window se richiesto
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        
        # Halo Attention (sostituisce Window Attention)
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        
        # MLP (stesso identico del originale)
        shortcut = x
        x = self.norm2(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        x = self.mlp(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = shortcut + x
        
        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        
        return x


class CompatibleEnhancedUFormer(nn.Module):
    """
    Enhanced UFormer COMPLETAMENTE COMPATIBILE con checkpoint esistente
    
    Strategia:
    - Carica tutti i pesi dal checkpoint originale
    - Sostituisce solo WindowAttention -> HaloWindowAttention  
    - Mantiene IDENTICA architettura encoder-decoder
    - Aggiunge shifted windows per alcuni blocchi
    """
    
    def __init__(self, 
                 embed_dim=96,
                 window_size=8, 
                 halo_size=4,  # Halo più piccolo per compatibilità
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 mlp_ratio=4.0,
                 patch_size=4):
        super().__init__()
        
        # IDENTICO al UFormer originale
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        
        # Patch embedding (IDENTICO)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Encoder layers (COMPATIBILI)
        self.encoder_layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer_depth = depths[i_layer]
            layer_heads = num_heads[i_layer]
            
            # Downsample (IDENTICO)
            if i_layer > 0:
                self.encoder_layers.append(
                    nn.Conv2d(int(embed_dim * 2 ** (i_layer-1)), layer_dim, 
                             kernel_size=2, stride=2)
                )
            
            # Enhanced blocks (COMPATIBILI - stesse dimensioni)
            layer_blocks = nn.ModuleList()
            for i_block in range(layer_depth):
                # Alterna shift per comunicazione cross-window
                shift_size = self.window_size // 2 if (i_block % 2 == 1) else 0
                
                layer_blocks.append(
                    EnhancedUFormerBlock(
                        dim=layer_dim,
                        window_size=window_size,
                        halo_size=halo_size,
                        num_heads=layer_heads,
                        mlp_ratio=mlp_ratio,
                        shift_size=shift_size
                    )
                )
            
            self.encoder_layers.append(layer_blocks)
        
        # Bottleneck (COMPATIBILE)
        bottleneck_dim = int(embed_dim * 2 ** (self.num_layers - 1))
        self.bottleneck = EnhancedUFormerBlock(
            dim=bottleneck_dim,
            window_size=window_size,
            halo_size=halo_size,
            num_heads=num_heads[-1],
            mlp_ratio=mlp_ratio,
            shift_size=0
        )
        
        # Decoder layers (IDENTICI per compatibilità)
        self.decoder_layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers - 1, -1, -1):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer_depth = depths[i_layer]
            layer_heads = num_heads[i_layer]
            
            # Upsample (IDENTICO)
            if i_layer < self.num_layers - 1:
                self.decoder_layers.append(
                    nn.ConvTranspose2d(int(embed_dim * 2 ** (i_layer+1)), layer_dim,
                                      kernel_size=2, stride=2)
                )
                # Skip connection fusion (IDENTICO)
                self.decoder_layers.append(
                    nn.Conv2d(layer_dim * 2, layer_dim, kernel_size=1)
                )
            
            # Decoder blocks (COMPATIBILI)
            layer_blocks = nn.ModuleList()
            for i_block in range(layer_depth):
                shift_size = self.window_size // 2 if (i_block % 2 == 1) else 0
                
                layer_blocks.append(
                    EnhancedUFormerBlock(
                        dim=layer_dim,
                        window_size=window_size,
                        halo_size=halo_size,
                        num_heads=layer_heads,
                        mlp_ratio=mlp_ratio,
                        shift_size=shift_size
                    )
                )
            
            self.decoder_layers.append(layer_blocks)
        
        # Output heads (IDENTICI)
        self.patch_unembed = nn.ConvTranspose2d(embed_dim, embed_dim, 
                                               kernel_size=patch_size, stride=patch_size)
        self.starless_head = nn.Conv2d(embed_dim, 3, kernel_size=3, padding=1)
        self.mask_head = nn.Conv2d(embed_dim, 1, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # IDENTICA implementazione del UFormer originale
        B, _, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Encoder con skip connections
        skip_connections = []
        
        layer_idx = 0
        for i_layer in range(self.num_layers):
            # Downsample se necessario
            if i_layer > 0:
                x = self.encoder_layers[layer_idx](x)
                layer_idx += 1
            
            # Skip connection
            skip_connections.append(x)
            
            # Enhanced blocks (con Halo Attention)
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
        
        # Output
        x = self.patch_unembed(x)
        starless = torch.sigmoid(self.starless_head(x))
        mask = torch.sigmoid(self.mask_head(x))
        
        return starless, mask
    
    def load_pretrained_compatible(self, checkpoint_path: str):
        """
        Carica checkpoint esistente in modo COMPLETAMENTE compatibile
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Carica tutto - DEVE essere completamente compatibile
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=True)
        
        print(f"Loaded pretrained model from {checkpoint_path}")
        print(f"Missing keys: {len(missing_keys)} (should be 0)")
        print(f"Unexpected keys: {len(unexpected_keys)} (should be 0)")
        
        if missing_keys or unexpected_keys:
            print("WARNING: Model not fully compatible!")
            print(f"Missing: {missing_keys[:5]}")
            print(f"Unexpected: {unexpected_keys[:5]}")
        
        return missing_keys, unexpected_keys

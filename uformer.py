"""
UFormer per rimozione stelle - SEMPLICE ed EFFICACE
Testa doppia: starless image + star mask
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


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


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Reshape to windows
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = x.view(B, H // self.window_size, self.window_size, 
                   W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size * self.window_size, C)  # [B*windows, window_size^2, C]
        
        # Attention
        qkv = self.qkv(x).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*windows, heads, window_size^2, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        x = self.proj(x)
        
        # Reshape back
        x = x.view(B, H // self.window_size, W // self.window_size, 
                   self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return x


class UFormerBlock(nn.Module):
    def __init__(self, dim, window_size=8, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Attention
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        
        # MLP
        shortcut = x
        x = self.norm2(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)  # [B*H*W, C]
        x = self.mlp(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        x = shortcut + x
        
        return x


class UFormerStarRemoval(nn.Module):
    """
    UFormer per rimozione stelle
    Testa doppia: predice sia immagine starless che maschera stelle
    """
    
    def __init__(self, 
                 in_channels: int = 3,
                 out_channels: int = 3,
                 embed_dim: int = 96,
                 depths: list = [2, 2, 6, 2],
                 num_heads: list = [3, 6, 12, 24],
                 window_size: int = 8):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        dims = [embed_dim * (2 ** i) for i in range(self.num_layers)]
        
        for i in range(self.num_layers):
            dim = dims[i]
            depth = depths[i]
            num_head = num_heads[i]
            
            # Encoder blocks
            blocks = nn.ModuleList([
                UFormerBlock(dim, window_size, num_head)
                for _ in range(depth)
            ])
            self.encoder_layers.append(blocks)
            
            # Downsampling (except last layer)
            if i < self.num_layers - 1:
                downsample = nn.Conv2d(dim, dims[i+1], kernel_size=2, stride=2)
                self.downsample_layers.append(downsample)
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        
        for i in range(self.num_layers - 1, 0, -1):
            dim = dims[i]
            prev_dim = dims[i-1]
            depth = depths[i-1]
            num_head = num_heads[i-1]
            
            # Upsampling
            upsample = nn.ConvTranspose2d(dim, prev_dim, kernel_size=2, stride=2)
            self.upsample_layers.append(upsample)
            
            # Decoder blocks (with skip connection)
            blocks = nn.ModuleList([
                UFormerBlock(prev_dim * 2, window_size, num_head)  # *2 for skip connection
                for _ in range(depth)
            ])
            self.decoder_layers.append(blocks)
            
            # Reduce channels after skip connection
            self.decoder_layers.append(nn.Conv2d(prev_dim * 2, prev_dim, kernel_size=1))
        
        # Final upsampling to original resolution
        self.final_upsample = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=4)
        
        # Testa doppia
        self.head_starless = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, out_channels, kernel_size=1)
        )
        
        self.head_mask = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1)
            # No activation - output logits for BCEWithLogitsLoss
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            tuple: (starless_image, star_mask)
                - starless_image: [B, 3, H, W]
                - star_mask: [B, 1, H, W]
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/4, W/4]
        
        # Encoder with skip connections
        skip_connections = []
        
        for i, (blocks, downsample) in enumerate(zip(self.encoder_layers[:-1], self.downsample_layers)):
            # Apply blocks
            for block in blocks:
                x = block(x)
            skip_connections.append(x)
            
            # Downsample
            x = downsample(x)
        
        # Bottleneck (last encoder layer)
        for block in self.encoder_layers[-1]:
            x = block(x)
        
        # Decoder with skip connections
        for i, (upsample, blocks, reduce_channels) in enumerate(zip(
            self.upsample_layers, 
            self.decoder_layers[::2], 
            self.decoder_layers[1::2]
        )):
            # Upsample
            x = upsample(x)
            
            # Skip connection
            skip = skip_connections[-(i+1)]
            x = torch.cat([x, skip], dim=1)
            
            # Apply blocks
            for block in blocks:
                x = block(x)
            
            # Reduce channels
            x = reduce_channels(x)
        
        # Final upsampling
        x = self.final_upsample(x)  # [B, embed_dim, H, W]
        
        # Testa doppia
        starless = self.head_starless(x)  # [B, 3, H, W]
        mask = self.head_mask(x)  # [B, 1, H, W]
        
        return starless, mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    model = UFormerStarRemoval()
    x = torch.randn(2, 3, 512, 512)
    
    starless, mask = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Starless: {starless.shape}")
    print(f"Mask: {mask.shape}")
    print(f"Parameters: {count_parameters(model):,}")

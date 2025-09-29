"""
Halo Attention implementation for UFormer star removal
Based on HAT (Hybrid Attention Transformer) for Image Restoration

Risolve i quadretti pixelati attorno alle stelle grandi permettendo
comunicazione tra finestre adiacenti tramite "halo" attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class HaloAttention(nn.Module):
    """
    Halo Attention: Q dal patch centrale, K/V dal patch + alone intorno
    
    Args:
        dim: Dimensione delle feature
        window_size: Dimensione della finestra centrale (16 per stelle grandi) 
        halo_size: Dimensione dell'alone intorno (8 per ws=16)
        num_heads: Numero di attention heads
        qkv_bias: Se usare bias in qkv projection
        attn_drop: Dropout rate per attention
        proj_drop: Dropout rate per output projection
        downsample_halo: Se fare downsampling di K/V nell'halo per efficienza
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int = 16,
        halo_size: int = 8, 
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        downsample_halo: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.halo_size = halo_size
        self.num_heads = num_heads
        self.downsample_halo = downsample_halo
        
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Q projection (solo per centro)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # K, V projections 
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size + 2 * halo_size - 1) ** 2, num_heads)
        )
        
        # Coordinate grids per relative position
        coords_h = torch.arange(-(halo_size), window_size + halo_size)
        coords_w = torch.arange(-(halo_size), window_size + halo_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size + halo_size - 1
        relative_coords[:, :, 1] += window_size + halo_size - 1
        relative_coords[:, :, 0] *= 2 * window_size + 2 * halo_size - 1
        
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        # Inizializzazione
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x: torch.Tensor, shift_size: int = 0) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, H, W, C)
            shift_size: Shifted window size per Swin-style shifting
        
        Returns:
            Output tensor (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # Shifted window partitioning se necessario
        if shift_size > 0:
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        
        # Window partitioning con halo
        windows_with_halo = self.window_partition_with_halo(x)  # (B*num_windows, ws+2*halo, ws+2*halo, C)
        
        # Separa centro e halo
        center_windows, halo_windows = self.separate_center_halo(windows_with_halo)
        
        # Compute Q solo dal centro
        q = self.q_proj(center_windows).reshape(
            -1, self.window_size * self.window_size, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)  # (B*num_windows, num_heads, ws*ws, head_dim)
        
        # Compute K, V da centro + halo
        if self.downsample_halo and self.halo_size > 0:
            # Downsample halo per efficienza
            halo_downsampled = F.avg_pool2d(
                halo_windows.permute(0, 3, 1, 2), 
                kernel_size=2, stride=2
            ).permute(0, 2, 3, 1)
            kv_input = torch.cat([center_windows, halo_downsampled.flatten(1, 2)], dim=1)
        else:
            kv_input = windows_with_halo.flatten(1, 2)
        
        k = self.k_proj(kv_input).reshape(
            -1, kv_input.shape[1], self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        
        v = self.v_proj(kv_input).reshape(
            -1, kv_input.shape[1], self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            (self.window_size + 2*self.halo_size)**2, 
            (self.window_size + 2*self.halo_size)**2, 
            -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        
        # Applica bias solo alla porzione rilevante
        center_size = self.window_size * self.window_size
        attn[:, :, :center_size, :] += relative_position_bias[:, 
            self.halo_size*2*self.window_size:self.halo_size*2*self.window_size+center_size,
            :
        ].unsqueeze(0)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention e riproietta
        x_attended = (attn @ v).transpose(1, 2).reshape(-1, center_size, C)
        x_attended = self.proj(x_attended)
        x_attended = self.proj_drop(x_attended)
        
        # Reshape per window reverse
        x_attended = x_attended.view(-1, self.window_size, self.window_size, C)
        
        # Window reverse
        x_output = self.window_reverse(x_attended, H, W)
        
        # Reverse shift se necessario
        if shift_size > 0:
            x_output = torch.roll(x_output, shifts=(shift_size, shift_size), dims=(1, 2))
            
        return x_output
    
    def window_partition_with_halo(self, x: torch.Tensor) -> torch.Tensor:
        """
        Partiziona immagine in finestre con halo
        """
        B, H, W, C = x.shape
        
        # Padding per halo (corretto)
        x_padded = F.pad(x.permute(0, 3, 1, 2), 
                        (self.halo_size, self.halo_size, self.halo_size, self.halo_size), 
                        mode='reflect').permute(0, 2, 3, 1)
        
        # Dimensioni dopo padding
        H_pad, W_pad = H + 2 * self.halo_size, W + 2 * self.halo_size
        
        # Window partition con halo
        ws_with_halo = self.window_size + 2 * self.halo_size
        
        # Calcola numero di finestre
        num_windows_h = H // self.window_size
        num_windows_w = W // self.window_size
        
        # Estrai finestre con halo (sliding windows)
        windows = []
        for i in range(num_windows_h):
            for j in range(num_windows_w):
                h_start = i * self.window_size
                h_end = h_start + ws_with_halo
                w_start = j * self.window_size  
                w_end = w_start + ws_with_halo
                
                window = x_padded[:, h_start:h_end, w_start:w_end, :]
                windows.append(window)
        
        # Stack windows
        x_windows = torch.stack(windows, dim=1)  # [B, num_windows, ws_with_halo, ws_with_halo, C]
        x_windows = x_windows.view(-1, ws_with_halo, ws_with_halo, C)  # [B*num_windows, ws_with_halo, ws_with_halo, C]
        
        return x_windows
    
    def separate_center_halo(self, windows_with_halo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Separa centro e halo dalle finestre
        """
        h = self.halo_size
        ws = self.window_size
        
        # Centro: [h:h+ws, h:h+ws]
        center = windows_with_halo[:, h:h+ws, h:h+ws, :].flatten(1, 2)
        
        # Halo: tutto il resto
        halo_mask = torch.ones_like(windows_with_halo[:, :, :, 0], dtype=torch.bool)
        halo_mask[:, h:h+ws, h:h+ws] = False
        halo = windows_with_halo[halo_mask].view(
            windows_with_halo.shape[0], -1, windows_with_halo.shape[-1]
        )
        
        return center, halo
    
    def window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Reverse window partition
        """
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(
            B, H // self.window_size, W // self.window_size,
            self.window_size, self.window_size, -1
        ).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        
        return x


class ShiftedHaloAttention(nn.Module):
    """
    Halo Attention con shifted windows (stile Swin Transformer)
    Alterna tra finestre normali e shifted per permettere comunicazione cross-window
    """
    
    def __init__(
        self, 
        dim: int,
        window_size: int = 16,
        halo_size: int = 8,
        num_heads: int = 8,
        shift: bool = False,
        **kwargs
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        
        self.attention = HaloAttention(
            dim=dim,
            window_size=window_size,
            halo_size=halo_size,
            num_heads=num_heads,
            **kwargs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x, shift_size=self.shift_size)

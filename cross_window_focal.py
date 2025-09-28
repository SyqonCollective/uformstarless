"""
Cross-Window Focal Attention for UFormer
Blocco che si inserisce ogni 2 blocchi per permettere comunicazione a lungo raggio

Caratteristiche:
- Q nella finestra corrente
- K/V da finestra + 1 anello di finestre vicine  
- Relative position bias per peso maggiore al centro
- Downsampling K/V dalle finestre lontane per efficienza
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class CrossWindowFocalAttention(nn.Module):
    """
    Focal Attention che permette comunicazione cross-window
    
    Args:
        dim: Dimensione delle feature
        window_size: Dimensione finestra base
        focal_window: Raggio di finestre vicine da includere (1 = 3x3 grid di finestre)
        num_heads: Numero attention heads
        focal_level: Numero di livelli di focus (1-3)
        focal_factor: Fattore downsampling per livelli lontani
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int = 16,
        focal_window: int = 1,  # 1 anello di finestre vicine
        num_heads: int = 8,
        focal_level: int = 2,
        focal_factor: int = 2,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.num_heads = num_heads
        
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias) 
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Focal weights per dare più peso al centro
        self.focal_layers = nn.ModuleList([
            nn.Linear(dim, dim, bias=qkv_bias) 
            for _ in range(focal_level)
        ])
        
        # Position encoding per distinguere centro vs vicini
        self.position_bias = nn.Parameter(torch.zeros(num_heads, (2*focal_window+1)**2))
        nn.init.trunc_normal_(self.position_bias, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input (B, H, W, C)
        Returns:
            Output (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # Window partition
        x_windows = self.window_partition(x)  # (B*nW, ws, ws, C)
        nW = x_windows.shape[0] // B
        
        # Gather neighboring windows per ogni finestra centrale
        focal_windows = self.gather_focal_windows(x, nW)  # (B*nW, num_focal_windows, ws_effective, ws_effective, C)
        
        # Q dal centro (finestra corrente)
        q = self.q_proj(x_windows.flatten(1, 2)).reshape(
            -1, self.window_size * self.window_size, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)  # (B*nW, num_heads, ws*ws, head_dim)
        
        # K, V da tutte le finestre focali (centro + vicine)  
        focal_flat = focal_windows.flatten(1, 3)  # (B*nW, total_focal_tokens, C)
        
        k = self.k_proj(focal_flat).reshape(
            -1, focal_flat.shape[1], self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        
        v = self.v_proj(focal_flat).reshape(
            -1, focal_flat.shape[1], self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add focal position bias (più peso al centro)
        attn = self.add_focal_bias(attn)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        # Reshape e reverse windows
        out = out.view(-1, self.window_size, self.window_size, C)
        out = self.window_reverse(out, H, W)
        
        return out
    
    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        """Partiziona in finestre"""
        B, H, W, C = x.shape
        x = x.view(B, H // self.window_size, self.window_size, 
                  W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        return windows
    
    def window_reverse(self, windows: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reverse window partitioning"""
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, 
                        self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    
    def gather_focal_windows(self, x: torch.Tensor, nW: int) -> torch.Tensor:
        """
        Raccoglie finestre focali (centro + vicine) per ogni finestra
        """
        B, H, W, C = x.shape
        nH, nW_grid = H // self.window_size, W // self.window_size
        
        # Pad per gestire bordi
        x_padded = F.pad(x, (0, 0, self.window_size, self.window_size, 
                            self.window_size, self.window_size), mode='reflect')
        
        focal_windows = []
        
        for i in range(nH):
            for j in range(nW_grid):
                window_focal = []
                
                # Raccoglie finestre in griglia (2*focal_window+1)x(2*focal_window+1)
                for di in range(-self.focal_window, self.focal_window + 1):
                    for dj in range(-self.focal_window, self.focal_window + 1):
                        # Coordinate finestra vicina
                        fi, fj = i + di, j + dj
                        
                        # Estrai finestra (con padding già applicato)
                        start_h = (fi + 1) * self.window_size  # +1 per il padding
                        start_w = (fj + 1) * self.window_size
                        
                        focal_window = x_padded[:, 
                            start_h:start_h + self.window_size,
                            start_w:start_w + self.window_size, 
                            :
                        ]
                        
                        # Downsample se non è la finestra centrale per efficienza
                        if abs(di) + abs(dj) > 0:  # Non centro
                            focal_window = F.avg_pool2d(
                                focal_window.permute(0, 3, 1, 2),
                                kernel_size=self.focal_factor, 
                                stride=self.focal_factor
                            ).permute(0, 2, 3, 1)
                        
                        window_focal.append(focal_window)
                
                # Stack finestre focali per questa posizione
                focal_windows.append(torch.stack(window_focal, dim=1))
        
        # Concatenate tutti
        focal_windows = torch.cat(focal_windows, dim=0)  # (B*nW, num_focal_windows, ws_eff, ws_eff, C)
        
        return focal_windows
    
    def add_focal_bias(self, attn: torch.Tensor) -> torch.Tensor:
        """
        Aggiunge bias per dare più peso al centro vs periferia
        """
        # Position bias: centro ha peso maggiore, bordi peso minore
        bias = self.position_bias.unsqueeze(0).unsqueeze(2)  # (1, num_heads, 1, num_focal_positions)
        
        # Replica il bias per ogni query token
        bias_expanded = bias.expand(attn.shape[0], -1, attn.shape[2], -1)
        
        # Centro (prima finestra) ha bias 0, altre hanno bias negativo
        center_tokens = self.window_size * self.window_size
        bias_expanded[:, :, :, :center_tokens] *= 0  # Centro: no penalty
        bias_expanded[:, :, :, center_tokens:] *= -1  # Vicini: penalty
        
        return attn + bias_expanded


class FocalTransformerBlock(nn.Module):
    """
    Transformer block con Cross-Window Focal Attention
    Si inserisce ogni 2 blocchi normali
    """
    
    def __init__(
        self,
        dim: int,
        window_size: int = 16,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        **focal_kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.focal_attn = CrossWindowFocalAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            attn_drop=attn_drop,
            **focal_kwargs
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Focal attention con residual
        x = x + self.focal_attn(self.norm1(x))
        
        # MLP con residual  
        x = x + self.mlp(self.norm2(x))
        
        return x

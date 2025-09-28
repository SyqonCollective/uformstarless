#!/usr/bin/env python3
"""
Test different window sizes for UFormer star removal
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

from uformer import UFormerStarRemoval

def test_window_sizes():
    """Test UFormer with different window sizes"""
    
    # Load test image
    image_path = input("Enter path to test image: ")
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Convert to tensor
    img_tensor = torch.FloatTensor(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    # Test different window sizes
    window_sizes = [8, 16, 32, 64]
    checkpoint_path = input("Enter checkpoint path: ")
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    for win_size in window_sizes:
        print(f"\nTesting window size: {win_size}")
        
        try:
            # Create model with different window size
            model = UFormerStarRemoval(
                embed_dim=96,
                window_size=win_size,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24]
            )
            
            # Load checkpoint (this might not work if architectures don't match)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                try:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"✓ Loaded checkpoint (some layers might be missing)")
                except Exception as e:
                    print(f"✗ Failed to load checkpoint: {e}")
                    continue
            
            model.to(device)
            model.eval()
            
            # Process image
            with torch.no_grad():
                # Resize to be compatible with window size
                h, w = img_tensor.shape[2], img_tensor.shape[3]
                new_h = ((h + win_size - 1) // win_size) * win_size
                new_w = ((w + win_size - 1) // win_size) * win_size
                
                # Pad image
                padded = torch.zeros(1, 3, new_h, new_w)
                padded[:, :, :h, :w] = img_tensor
                
                result = model(padded.to(device))
                
                if isinstance(result, tuple):
                    processed = result[0]  # starless image
                else:
                    processed = result
                
                # Crop back to original size
                processed = processed[:, :, :h, :w]
                
                # Convert to PIL
                processed_np = processed.cpu().squeeze(0).permute(1, 2, 0).numpy()
                processed_np = np.clip(processed_np * 255, 0, 255).astype(np.uint8)
                result_img = Image.fromarray(processed_np)
                
                # Save result
                output_path = f"test_window_{win_size}.png"
                result_img.save(output_path)
                print(f"✓ Saved: {output_path}")
                
        except Exception as e:
            print(f"✗ Error with window size {win_size}: {e}")

if __name__ == "__main__":
    test_window_sizes()

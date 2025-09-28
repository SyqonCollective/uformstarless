#!/usr/bin/env python3
"""
Quick UFormer inference script - uso semplice
"""

import sys
from pathlib import Path
from inference_uformer import UFormerInference

def quick_inference():
    """Inference veloce con parametri di default"""
    
    if len(sys.argv) < 4:
        print("Usage: python quick_inference.py <checkpoint> <input_image> <output_image>")
        print("Example: python quick_inference.py best_model.pth input.jpg output_starless.jpg")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Verifica che il checkpoint esista
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Verifica che l'input esista
    if not Path(input_path).exists():
        print(f"Error: Input image not found: {input_path}")
        sys.exit(1)
    
    print("🚀 UFormer Star Removal - Quick Inference")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Initialize inference con parametri ottimali
    inference = UFormerInference(
        checkpoint_path=checkpoint_path,
        device="auto",          # Auto-detect GPU/CPU
        tile_size=512,          # Tile size ottimale
        overlap=64              # Overlap per smooth blending
    )
    
    # Process image
    print("\n🔄 Processing...")
    starless_image = inference.process_image(input_path)
    
    # Save result
    starless_image.save(output_path)
    
    print(f"✅ Done! Starless image saved to: {output_path}")

if __name__ == "__main__":
    quick_inference()

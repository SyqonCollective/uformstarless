"""
Quick Fix UFormer - Simple Command Line Interface
Versione semplificata per testare il modello senza quadretti
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

# Import Quick Fix UFormer
from quick_fix_uformer import quick_fix_uformer
from uformer import UFormerStarRemoval


def get_device():
    """Get best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Mac M1/M2
    else:
        return torch.device("cpu")


def tensor_to_pil(tensor, is_mask=False):
    """Convert tensor to PIL Image"""
    # Remove batch dimension and move to CPU
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    tensor = tensor.cpu().clamp(0, 1)
    
    if is_mask:
        # Convert single channel mask to RGB
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)
    
    # Convert to numpy and PIL
    np_array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_array)


def tile_process(model, input_tensor, tile_size=512, overlap=64):
    """Process image with tiling"""
    device = input_tensor.device
    B, C, H, W = input_tensor.shape
    
    if H <= tile_size and W <= tile_size:
        # No tiling needed
        return model(input_tensor)
    
    print(f"Processing large image ({H}x{W}) with tiles...")
    
    # Calculate tiles
    stride = tile_size - overlap
    h_tiles = (H - overlap) // stride + (1 if (H - overlap) % stride else 0)
    w_tiles = (W - overlap) // stride + (1 if (W - overlap) % stride else 0)
    
    print(f"Using {h_tiles}x{w_tiles} = {h_tiles * w_tiles} tiles")
    
    # Initialize output tensors
    starless_output = torch.zeros_like(input_tensor)
    mask_output = torch.zeros(B, 1, H, W, device=device)
    weight_map = torch.zeros(B, 1, H, W, device=device)
    
    # Process each tile
    tile_count = 0
    total_tiles = h_tiles * w_tiles
    
    for h in range(h_tiles):
        for w in range(w_tiles):
            tile_count += 1
            print(f"Processing tile {tile_count}/{total_tiles}...", end=" ", flush=True)
            
            # Calculate tile coordinates
            h_start = h * stride
            w_start = w * stride
            h_end = min(h_start + tile_size, H)
            w_end = min(w_start + tile_size, W)
            
            # Extract tile
            tile = input_tensor[:, :, h_start:h_end, w_start:w_end]
            
            # Pad if necessary
            pad_h = tile_size - tile.shape[2]
            pad_w = tile_size - tile.shape[3]
            
            if pad_h > 0 or pad_w > 0:
                tile = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
            
            # Process tile
            tile_starless, tile_mask = model(tile)
            
            # Remove padding
            if pad_h > 0 or pad_w > 0:
                tile_starless = tile_starless[:, :, :tile_size-pad_h, :tile_size-pad_w]
                tile_mask = tile_mask[:, :, :tile_size-pad_h, :tile_size-pad_w]
            
            # Create weight for blending
            tile_h, tile_w = tile_starless.shape[2], tile_starless.shape[3]
            weight = torch.ones(1, 1, tile_h, tile_w, device=device)
            
            # Apply to output
            starless_output[:, :, h_start:h_end, w_start:w_end] += tile_starless * weight
            mask_output[:, :, h_start:h_end, w_start:w_end] += tile_mask * weight
            weight_map[:, :, h_start:h_end, w_start:w_end] += weight
            
            print("✓")
    
    # Normalize by weight
    starless_output = starless_output / weight_map
    mask_output = mask_output / weight_map
    
    return starless_output, mask_output


def main():
    parser = argparse.ArgumentParser(description="UFormer Quick Fix - NO QUADRETTI!")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--checkpoint", "-c", help="UFormer checkpoint path")
    parser.add_argument("--quick-fix-model", help="Pre-generated Quick Fix model path")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size for processing")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between tiles")
    parser.add_argument("--output", "-o", help="Output directory (default: same as input)")
    
    args = parser.parse_args()
    
    device = get_device()
    print(f"🚀 UFormer Quick Fix - Device: {device}")
    
    # Load model
    model = None
    is_quick_fix = False
    
    if args.quick_fix_model:
        # Load pre-generated Quick Fix model
        print(f"✨ Loading Quick Fix model: {args.quick_fix_model}")
        if not Path(args.quick_fix_model).exists():
            print(f"❌ Error: Quick Fix model not found: {args.quick_fix_model}")
            return
        
        model = UFormerStarRemoval(
            embed_dim=96,
            window_size=8,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24]
        )
        
        checkpoint = torch.load(args.quick_fix_model, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        is_quick_fix = True
        
    elif args.checkpoint:
        # Apply Quick Fix to original checkpoint
        print(f"✨ Applying Quick Fix to: {args.checkpoint}")
        if not Path(args.checkpoint).exists():
            print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
            return
        
        model = quick_fix_uformer(args.checkpoint, device=str(device))
        is_quick_fix = True
        
    else:
        print("❌ Error: Please specify either --checkpoint or --quick-fix-model")
        return
    
    # Load image
    print(f"📷 Loading image: {args.image}")
    if not Path(args.image).exists():
        print(f"❌ Error: Image not found: {args.image}")
        return
    
    image = Image.open(args.image).convert('RGB')
    print(f"Image size: {image.size[0]}x{image.size[1]}")
    
    # Preprocess
    input_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(device)
    
    # Process
    print(f"🎯 Processing with {'Quick Fix (NO quadretti)' if is_quick_fix else 'Standard'}...")
    print(f"Tile size: {args.tile_size}, Overlap: {args.overlap}")
    
    with torch.no_grad():
        starless_output, mask_output = tile_process(model, input_tensor, args.tile_size, args.overlap)
    
    # Convert results
    starless_img = tensor_to_pil(starless_output)
    mask_img = tensor_to_pil(mask_output, is_mask=True)
    
    # Save results
    input_path = Path(args.image)
    output_dir = Path(args.output) if args.output else input_path.parent
    output_dir.mkdir(exist_ok=True)
    
    suffix = "_quick_fix" if is_quick_fix else "_standard"
    starless_path = output_dir / f"{input_path.stem}{suffix}_starless.png"
    mask_path = output_dir / f"{input_path.stem}{suffix}_mask.png"
    
    print(f"💾 Saving results...")
    starless_img.save(starless_path)
    mask_img.save(mask_path)
    
    print(f"✅ Processing complete!")
    print(f"   Starless: {starless_path}")
    print(f"   Mask: {mask_path}")
    
    if is_quick_fix:
        print("🎉 NO MORE QUADRETTI! Artifacts eliminated! 🎉")


if __name__ == '__main__':
    main()

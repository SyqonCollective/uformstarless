#!/usr/bin/env python3
"""
Script demo per Enhanced UFormer con shifted windows
Mostra come usare il modello migliorato per eliminare quadretti 8x8

Caratteristiche:
✅ Shifted windows (shifted_window=True) per eliminare quadretti  
✅ Window size 8 per inferenza leggera
✅ Perceptual + L1 + SSIM loss per qualità visiva
✅ Halo attention per comunicazione cross-window
✅ Focal blocks per stelle giganti

Uso:
    python demo_enhanced_shifted.py --config config_uformer.yaml --checkpoint best_model.pth --input image.jpg
"""

import argparse
import torch
import yaml
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

from enhanced_uformer import EnhancedUFormerStarRemoval


def load_config(config_path: str) -> dict:
    """Carica configurazione da YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint_path: str = None) -> EnhancedUFormerStarRemoval:
    """
    Carica Enhanced UFormer con shifted windows abilitati
    """
    model_config = config['model']
    
    model = EnhancedUFormerStarRemoval(
        embed_dim=model_config['embed_dim'],
        window_size=model_config['win_size'],  # 8 per inferenza leggera
        halo_size=model_config.get('halo_size', 4),
        depths=model_config['depths'],
        num_heads=model_config['num_heads'],
        focal_interval=model_config.get('focal_interval', 2),
        shifted_window=model_config.get('shifted_window', True)  # ESSENZIALE!
    )
    
    # Carica checkpoint se specificato
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"🔄 Loading checkpoint: {checkpoint_path}")
        missing_keys, unexpected_keys = model.load_pretrained_compatible(
            checkpoint_path, strict=False
        )
        print(f"✅ Loaded checkpoint")
        print(f"📦 New modules (from scratch): {len(missing_keys)}")
        if missing_keys:
            print(f"   - Examples: {missing_keys[:3]}")
    else:
        print("⚠️  No checkpoint loaded - using random initialization")
    
    return model


def preprocess_image(image_path: str) -> torch.Tensor:
    """Preprocessa immagine per inferenza"""
    image = Image.open(image_path).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # [1, 3, H, W]
    return image_tensor


def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    """Converte tensor in PIL Image"""
    # tensor: [1, 3, H, W] in [0, 1]
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)
    tensor = (tensor * 255).byte()
    
    # Convert to PIL
    tensor_np = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(tensor_np)


def inference(model: EnhancedUFormerStarRemoval, image_tensor: torch.Tensor, device: str = 'cpu') -> tuple:
    """
    Esegue inferenza con Enhanced UFormer
    
    Returns:
        starless_image: Immagine senza stelle
        star_mask: Maschera delle stelle rimosse
    """
    model.eval()
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        # Inferenza: dual output
        starless_pred, mask_pred = model(image_tensor)
        
    return starless_pred, mask_pred


def main():
    parser = argparse.ArgumentParser(description="Enhanced UFormer Demo con Shifted Windows")
    parser.add_argument('--config', required=True, help='Path al file config YAML')
    parser.add_argument('--checkpoint', help='Path al checkpoint (opzionale)')
    parser.add_argument('--input', required=True, help='Path immagine input')
    parser.add_argument('--output', help='Path output (default: input_starless.jpg)')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"🖥️  Using device: {device}")
    
    # Output path
    if not args.output:
        input_path = Path(args.input)
        args.output = input_path.parent / f"{input_path.stem}_starless{input_path.suffix}"
    
    print("🚀 Enhanced UFormer Demo - Eliminazione Quadretti 8x8")
    print("=" * 60)
    
    # Carica config
    print(f"📋 Loading config: {args.config}")
    config = load_config(args.config)
    
    # Print configuration
    model_config = config['model']
    print("📊 Model Configuration:")
    print(f"   - Window Size: {model_config['win_size']} (leggero per inferenza)")
    print(f"   - Shifted Windows: {model_config.get('shifted_window', True)} ✅ (elimina quadretti!)")
    print(f"   - Embed Dim: {model_config['embed_dim']}")
    print(f"   - Depths: {model_config['depths']} (profondo al centro)")
    print(f"   - Halo Size: {model_config.get('halo_size', 4)} (cross-window communication)")
    
    # Carica modello
    print(f"\\n🔧 Loading Enhanced UFormer...")
    model = load_model(config, args.checkpoint)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📈 Total parameters: {total_params:,}")
    
    # Preprocessa immagine
    print(f"\\n🖼️  Processing image: {args.input}")
    image_tensor = preprocess_image(args.input)
    H, W = image_tensor.shape[2], image_tensor.shape[3]
    print(f"   - Resolution: {W}x{H}")
    
    # Inferenza
    print("🔮 Running inference...")
    start_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    
    if device == 'cuda':
        start_time.record()
    
    starless_pred, mask_pred = inference(model, image_tensor, device)
    
    if device == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        inference_time = start_time.elapsed_time(end_time) / 1000  # seconds
    else:
        inference_time = 0.0  # Placeholder per CPU
    
    print(f"✅ Inference completed in {inference_time:.3f}s")
    
    # Postprocessing e salvataggio
    print(f"💾 Saving results...")
    
    # Starless image
    starless_image = postprocess_image(starless_pred)
    starless_image.save(args.output)
    print(f"   - Starless image: {args.output}")
    
    # Star mask (opzionale)
    mask_path = Path(args.output).parent / f"{Path(args.output).stem}_mask.jpg"
    mask_image = postprocess_image(mask_pred.repeat(1, 3, 1, 1))  # Convert to RGB
    mask_image.save(mask_path)
    print(f"   - Star mask: {mask_path}")
    
    print("\\n🎉 Demo completato!")
    print("\\n📝 Note:")
    print("   ✅ Shifted windows abilitati → niente più quadretti 8x8!")
    print("   ✅ Window size 8 → inferenza veloce")
    print("   ✅ Halo attention → comunicazione cross-window")
    print("   ✅ Focal blocks → gestione stelle giganti")
    print("\\n💡 Per training con perceptual loss, usa enhanced_uformer_finetune.py")


if __name__ == "__main__":
    main()
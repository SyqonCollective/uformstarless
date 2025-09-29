#!/usr/bin/env python3
"""
Debug loss components per identificare il problema
"""

import torch
import yaml
from star_dataset import create_dataloader
from enhanced_loss import EnhancedUFormerLoss

def debug_loss_components():
    """Debug dei valori di loss"""
    
    # Load config
    with open('config_5090_optimized.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataloader
    train_loader = create_dataloader(
        input_dir=config['data']['train_input_dir'],
        target_dir=config['data']['train_target_dir'],
        batch_size=2,  # Piccolo batch per debug
        num_workers=0,
        shuffle=False,
        is_training=True
    )
    
    # Setup loss
    loss_config = config.get('loss', {})
    criterion = EnhancedUFormerLoss(
        l1_weight=loss_config.get('l1_weight', 1.0),
        perceptual_weight=loss_config.get('perceptual_weight', 0.05),
        ssim_weight=loss_config.get('ssim_weight', 0.05),
        mask_weight=loss_config.get('mask_weight', 0.1),
        use_ssim=loss_config.get('use_ssim', True)
    ).to(device)
    
    print(f"Loss weights:")
    print(f"  L1: {criterion.l1_weight}")
    print(f"  Perceptual: {criterion.perceptual_weight}")
    print(f"  SSIM: {criterion.ssim_weight}")
    print(f"  Mask: {criterion.mask_weight}")
    
    # Test first batch
    for batch_idx, batch in enumerate(train_loader):
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        masks = batch['mask'].to(device)
        
        print(f"\nBatch {batch_idx}:")
        print(f"Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
        print(f"Target range: [{targets.min():.4f}, {targets.max():.4f}]")
        print(f"Mask range: [{masks.min():.4f}, {masks.max():.4f}]")
        
        # Simula predictions (usa target come pred per test)
        pred_starless = targets + torch.randn_like(targets) * 0.01  # Piccola differenza
        pred_mask = masks + torch.randn_like(masks) * 0.01
        
        # Test loss components
        with torch.no_grad():
            total_loss, loss_dict = criterion(pred_starless, pred_mask, targets, masks)
            
            print(f"\nLoss Components:")
            for key, value in loss_dict.items():
                print(f"  {key}: {value:.6f}")
            print(f"Total Loss: {total_loss.item():.6f}")
            
        break  # Solo primo batch
    
    print("\n✅ Debug completed!")

if __name__ == "__main__":
    debug_loss_components()
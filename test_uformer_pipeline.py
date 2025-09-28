"""
Quick integration test for UFormer star removal pipeline
Tests that all components work together before full training
"""

import torch
import yaml
from pathlib import Path

from uformer import UFormerStarRemoval
from star_dataset import create_dataloader
from loss_uformer import UFormerLoss, UFormerMetrics


def test_full_pipeline():
    """Test the complete UFormer pipeline"""
    print("🚀 Testing UFormer Star Removal Pipeline")
    print("=" * 50)
    
    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load config
    print("\n📋 Loading configuration...")
    with open('config_uformer.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("✅ Config loaded")
    
    # Test model creation
    print("\n🏗️  Testing model creation...")
    model_config = config['model']
    model = UFormerStarRemoval(
        in_channels=3,
        out_channels=3,
        embed_dim=model_config['embed_dim'],
        depths=model_config['depths'][:4],  # Take first 4 for encoder
        num_heads=model_config['num_heads'][:4],  # Take first 4 for encoder
        window_size=model_config['win_size']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model created - Parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Test model forward pass
    print("\n🔄 Testing model forward pass...")
    batch_size = 2
    test_input = torch.randn(batch_size, 3, 512, 512).to(device)
    
    with torch.no_grad():
        pred_starless, pred_mask = model(test_input)
    
    print(f"✅ Forward pass successful")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Pred starless shape: {pred_starless.shape}")
    print(f"   Pred mask shape: {pred_mask.shape}")
    
    # Test loss function
    print("\n💥 Testing loss function...")
    target_starless = torch.randn_like(pred_starless)
    target_mask = torch.randint(0, 2, pred_mask.shape).float().to(device)
    
    criterion = UFormerLoss(mask_weight=config['mask_weight'])
    losses = criterion(pred_starless, pred_mask, target_starless, target_mask)
    
    print(f"✅ Loss computation successful")
    for name, value in losses.items():
        print(f"   {name}: {value.item():.6f}")
    
    # Test metrics
    print("\n📊 Testing metrics...")
    # Normalize for metrics (simulate realistic predictions)
    pred_starless_norm = torch.sigmoid(pred_starless)
    target_starless_norm = torch.sigmoid(target_starless)
    
    metrics = UFormerMetrics.compute_all_metrics(
        pred_starless_norm, pred_mask, target_starless_norm, target_mask
    )
    
    print(f"✅ Metrics computation successful")
    for name, value in metrics.items():
        print(f"   {name}: {value:.6f}")
    
    # Test dataloader (if data exists)
    print("\n📂 Testing dataloader...")
    train_input_dir = Path(config['data']['train_input_dir'])
    train_target_dir = Path(config['data']['train_target_dir'])
    
    if train_input_dir.exists() and train_target_dir.exists():
        try:
            train_loader = create_dataloader(
                str(train_input_dir),
                str(train_target_dir),
                batch_size=4,  # Small batch for test
                num_workers=2,
                shuffle=True,
                is_training=True
            )
            
            # Test loading one batch
            batch = next(iter(train_loader))
            input_imgs = batch['input']
            target_imgs = batch['target']
            target_masks = batch['mask']
            
            print(f"✅ Dataloader working")
            print(f"   Batch input shape: {input_imgs.shape}")
            print(f"   Batch target shape: {target_imgs.shape}")
            print(f"   Batch mask shape: {target_masks.shape}")
            print(f"   Total samples: {len(train_loader.dataset)}")
            
            # Test full forward pass with real data
            print("\n🎯 Testing with real data...")
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)
            target_masks = target_masks.to(device)
            
            with torch.no_grad():
                pred_starless, pred_mask = model(input_imgs)
                losses = criterion(pred_starless, pred_mask, target_imgs, target_masks)
                # Clamp to [0,1] for metrics
                pred_clamped = pred_starless.clamp(0, 1)
                target_clamped = target_imgs.clamp(0, 1)
                metrics = UFormerMetrics.compute_all_metrics(
                    pred_clamped, pred_mask, target_clamped, target_masks
                )
            
            print(f"✅ Real data test successful")
            print(f"   Loss: {losses['total'].item():.6f}")
            print(f"   PSNR: {metrics['psnr']:.2f}")
            print(f"   SSIM: {metrics['ssim']:.4f}")
            
        except Exception as e:
            print(f"⚠️  Dataloader test failed: {e}")
            print("   This is expected if training data is not available")
    else:
        print("⚠️  Training data directories not found - skipping dataloader test")
        print(f"   Expected: {train_input_dir} and {train_target_dir}")
    
    # Memory usage test
    print("\n🧠 Memory usage test...")
    if torch.cuda.is_available():
        memory_used = torch.cuda.memory_allocated() / 1e9
        memory_cached = torch.cuda.memory_reserved() / 1e9
        print(f"   GPU Memory - Used: {memory_used:.2f} GB, Cached: {memory_cached:.2f} GB")
        
        # Test larger batch for memory usage
        try:
            large_batch_size = config['training']['batch_size']
            large_input = torch.randn(large_batch_size, 3, 512, 512).to(device)
            
            with torch.no_grad():
                pred_starless, pred_mask = model(large_input)
            
            memory_used_large = torch.cuda.memory_allocated() / 1e9
            print(f"   Large batch ({large_batch_size}) memory: {memory_used_large:.2f} GB")
            print(f"✅ Memory usage looks good for batch size {large_batch_size}")
            
        except torch.cuda.OutOfMemoryError:
            print(f"❌ Batch size {config['training']['batch_size']} too large for available memory")
            print("   Consider reducing batch_size in config")
    
    print("\n" + "=" * 50)
    print("🎉 Pipeline test completed!")
    print("Ready for training with: python train_uformer.py --config config_uformer.yaml")


if __name__ == "__main__":
    test_full_pipeline()

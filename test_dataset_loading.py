#!/usr/bin/env python3
"""
Test dataset loading per verificare compatibilità
"""

import torch
from star_dataset import StarRemovalDataset, create_dataloader
import yaml

def test_dataset_loading():
    """Test che il dataset si carichi correttamente"""
    
    # Load config
    with open('config_uformer.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Testing dataset loading...")
    
    # Create test dataloader
    train_loader = create_dataloader(
        input_dir=config['data']['train_input_dir'],
        target_dir=config['data']['train_target_dir'],
        batch_size=2,
        num_workers=0,  # Per test usiamo 0 workers
        shuffle=False,
        is_training=True
    )
    
    print(f"Dataset size: {len(train_loader.dataset)}")
    print(f"Number of batches: {len(train_loader)}")
    
    # Test first batch
    for batch_idx, batch in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"Batch type: {type(batch)}")
        print(f"Batch keys: {batch.keys()}")
        
        inputs = batch['input']
        targets = batch['target'] 
        masks = batch['mask']
        
        print(f"Input shape: {inputs.shape}")
        print(f"Target shape: {targets.shape}")
        print(f"Mask shape: {masks.shape}")
        print(f"Filenames: {batch['filename']}")
        
        break  # Solo primo batch per test
    
    print("\n✅ Dataset loading test passed!")

if __name__ == "__main__":
    test_dataset_loading()
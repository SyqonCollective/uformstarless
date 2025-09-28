#!/bin/bash
# Fix dataset e riavvia training

echo "🔧 Fixing dataset bug on RunPod..."

# Stop current training
pkill -f train_uformer.py

# Backup del file originale
cp star_dataset.py star_dataset.py.bak

cat > star_dataset.py << 'EOF'
"""
Dataset SEMPLICE per rimozione stelle
Carica input/target 512x512 senza complicazioni
"""

import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from typing import Dict, Optional, Tuple


class StarRemovalDataset(Dataset):
    """Dataset semplice per rimozione stelle"""
    
    def __init__(self, 
                 input_dir: str = None,
                 target_dir: str = None,
                 data_dir: str = None,
                 split: str = 'train',
                 augment: bool = True):
        """
        Args:
            input_dir: Directory with input images (if specified, ignores data_dir/split)
            target_dir: Directory with target images (if specified, ignores data_dir/split)  
            data_dir: Root directory containing train/val folders (legacy mode)
            split: 'train' or 'val' (only used with data_dir)
            augment: Apply light augmentations
        """
        self.augment = augment
        
        # Handle different initialization modes
        if input_dir and target_dir:
            # Direct mode: specific input/target directories
            self.input_dir = Path(input_dir)
            self.target_dir = Path(target_dir)
        else:
            # Legacy mode: data_dir with train/val structure  
            self.data_dir = Path(data_dir)
            self.split = split
            self.input_dir = self.data_dir / split / 'input'
            self.target_dir = self.data_dir / split / 'target'
        
        # Find all images
        self.image_files = []
        input_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        
        for ext in input_extensions:
            self.image_files.extend(list(self.input_dir.glob(f'*{ext}')))
            self.image_files.extend(list(self.input_dir.glob(f'*{ext.upper()}')))
        
        self.image_files = sorted(self.image_files)
        
        split_name = split if data_dir else "direct"
        print(f"Found {len(self.image_files)} images in {split_name}")
        
        # Augmentations LEGGERE - solo geometriche
        if self.augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.2)
            ], additional_targets={'target_image': 'image'})  # target_image gets same transforms as image
        else:
            self.transform = A.Compose([])  # No transforms, just pass through
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_path = self.image_files[idx]
        
        # Target path (stesso nome nella target directory)
        target_path = self.target_dir / input_path.name
        
        # Load input
        input_img = Image.open(input_path).convert('RGB')
        input_array = np.array(input_img).astype(np.float32) / 255.0
        
        # Load target
        if not target_path.exists():
            raise FileNotFoundError(f"Target not found for {input_path.name}")
        
        target_img = Image.open(target_path).convert('RGB')
        target_array = np.array(target_img).astype(np.float32) / 255.0
        
        # Genera maschera stelle (semplice: dove input != target)
        mask_array = np.mean(np.abs(input_array - target_array), axis=2, keepdims=True)
        mask_array = (mask_array > 0.01).astype(np.float32)  # Soglia per stelle
        
        # Augmentations
        if self.augment:
            # Usa additional_targets per target image, mask come mask
            transformed = self.transform(
                image=input_array,
                target_image=target_array,  # Come additional target
                mask=mask_array.squeeze(-1)  # Remove channel dim for albumentations mask
            )
            input_tensor = torch.from_numpy(transformed['image'].transpose(2, 0, 1))
            target_tensor = torch.from_numpy(transformed['target_image'].transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(transformed['mask']).unsqueeze(0)  # Add channel back
        else:
            input_tensor = torch.from_numpy(input_array.transpose(2, 0, 1))
            target_tensor = torch.from_numpy(target_array.transpose(2, 0, 1))
            mask_tensor = torch.from_numpy(mask_array.transpose(2, 0, 1))
        
        return {
            'input': input_tensor,        # [3, 512, 512]
            'target': target_tensor,      # [3, 512, 512] 
            'mask': mask_tensor,          # [1, 512, 512]
            'filename': input_path.name
        }


def create_dataloader(input_dir: str, target_dir: str,
                     batch_size: int = 16,
                     num_workers: int = 8,
                     shuffle: bool = True,
                     is_training: bool = True) -> torch.utils.data.DataLoader:
    """Create a single dataloader for input/target pairs"""
    
    dataset = StarRemovalDataset(
        input_dir=input_dir,
        target_dir=target_dir,
        augment=is_training
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_training
    )


def create_dataloaders(data_dir: str,
                      batch_size: int = 16,
                      num_workers: int = 8,
                      pin_memory: bool = True) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Crea train e validation dataloaders"""
    
    train_dataset = StarRemovalDataset(data_dir, 'train', augment=True)
    val_dataset = StarRemovalDataset(data_dir, 'val', augment=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing dataset...")
    
    # Test dataset creation
    dataset = StarRemovalDataset(
        input_dir="train/input",
        target_dir="train/target", 
        augment=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        for key, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                print(f"{key}: {tensor.shape} ({tensor.dtype})")
            else:
                print(f"{key}: {tensor}")
EOF

echo "✅ Dataset fixed! Restarting training..."

# Restart training
nohup python train_uformer.py --config config_uformer.yaml > uformer_train.log 2>&1 &

echo "📊 Training restarted. Monitor with:"
echo "tail -f uformer_train.log"

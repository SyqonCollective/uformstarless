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
        
        # Find images and filter only those with matching targets
        all_input_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
            all_input_files.extend(list(self.input_dir.glob(f'*{ext}')))
            all_input_files.extend(list(self.input_dir.glob(f'*{ext.upper()}')))
        
        # Filter only files that have corresponding targets
        self.image_files = []
        skipped_count = 0
        
        for input_file in all_input_files:
            target_file = self.target_dir / input_file.name
            if target_file.exists():
                self.image_files.append(input_file)
            else:
                skipped_count += 1
        
        self.image_files = sorted(self.image_files)
        split_name = split if data_dir else "direct"
        print(f"Found {len(self.image_files)} valid pairs in {split_name}")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} files without matching targets")
        
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
        target_path = self.target_dir / input_path.name
        
        # Load images (target guaranteed to exist due to filtering)
        input_img = Image.open(input_path).convert('RGB')
        input_array = np.array(input_img).astype(np.float32) / 255.0
        
        target_img = Image.open(target_path).convert('RGB')
        target_array = np.array(target_img).astype(np.float32) / 255.0
        
        # Generate mask automatically from difference
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
    # Test dataset
    dataset = StarRemovalDataset(".", 'train')
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Input shape: {sample['input'].shape}")
        print(f"Target shape: {sample['target'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Filename: {sample['filename']}")
        print(f"Mask mean: {sample['mask'].mean():.4f}")

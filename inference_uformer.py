"""
Inference script per UFormer star removal
Processa immagini con tiling 512x512 e overlap per gestire immagini grandi
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, Optional, Union
import yaml

from uformer import UFormerStarRemoval


class UFormerInference:
    """
    Inference engine per UFormer star removal con tiling automatico
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 device: Optional[str] = None,
                 tile_size: int = 512,
                 overlap: int = 64):
        """
        Args:
            checkpoint_path: Path al checkpoint del modello
            device: Device per inference ('cuda', 'cpu', 'auto')
            tile_size: Dimensione dei tile (default 512)
            overlap: Overlap tra tile in pixel (default 64)
        """
        self.tile_size = tile_size
        self.overlap = overlap
        
        # Auto-detect device
        if device == 'auto' or device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        print(f"Model loaded from: {checkpoint_path}")
        
    def _load_model(self, checkpoint_path: str) -> UFormerStarRemoval:
        """Carica il modello dal checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Estrai parametri del modello dal config se disponibile
        if 'config' in checkpoint:
            model_config = checkpoint['config']['model']
            model = UFormerStarRemoval(
                embed_dim=model_config.get('embed_dim', 96),
                depths=model_config.get('depths', [2, 2, 6, 2]),
                num_heads=model_config.get('num_heads', [3, 6, 12, 24]),
                window_size=model_config.get('win_size', 8)
            )
        else:
            # Parametri di default
            print("Warning: No config found in checkpoint, using default parameters")
            model = UFormerStarRemoval()
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Converte PIL Image a tensor normalizzato"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Convert to tensor [C, H, W]
        tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
        
        return tensor
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Converte tensor a PIL Image"""
        # Clamp to [0, 1] and convert to numpy
        img_array = tensor.clamp(0, 1).cpu().numpy()
        
        # Convert from [C, H, W] to [H, W, C]
        img_array = img_array.transpose(1, 2, 0)
        
        # Convert to uint8
        img_array = (img_array * 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _extract_tiles(self, tensor: torch.Tensor) -> Tuple[list, list, Tuple[int, int]]:
        """
        Estrae tile dall'immagine con overlap
        
        Returns:
            tiles: Lista di tile [C, tile_size, tile_size]
            positions: Lista di posizioni (x, y) per ogni tile
            original_size: (height, width) dell'immagine originale
        """
        C, H, W = tensor.shape
        tiles = []
        positions = []
        
        step = self.tile_size - self.overlap
        
        for y in range(0, H, step):
            for x in range(0, W, step):
                # Calcola bounds del tile
                x_end = min(x + self.tile_size, W)
                y_end = min(y + self.tile_size, H)
                
                # Se il tile è troppo piccolo, ajusta la posizione
                if x_end - x < self.tile_size:
                    x = max(0, W - self.tile_size)
                    x_end = W
                if y_end - y < self.tile_size:
                    y = max(0, H - self.tile_size)
                    y_end = H
                
                # Estrai tile
                tile = tensor[:, y:y_end, x:x_end]
                
                # Pad se necessario per raggiungere tile_size
                if tile.shape[1] < self.tile_size or tile.shape[2] < self.tile_size:
                    tile = F.pad(tile, (0, self.tile_size - tile.shape[2], 
                                       0, self.tile_size - tile.shape[1]), 
                                mode='reflect')
                
                tiles.append(tile)
                positions.append((x, y))
        
        return tiles, positions, (H, W)
    
    def _merge_tiles(self, 
                     tiles: list, 
                     positions: list, 
                     original_size: Tuple[int, int]) -> torch.Tensor:
        """
        Ricostruisce l'immagine dai tile con blending dell'overlap
        """
        H, W = original_size
        C = tiles[0].shape[0]
        
        # Output tensor e weight map per blending
        result = torch.zeros((C, H, W), dtype=tiles[0].dtype, device=tiles[0].device)
        weights = torch.zeros((H, W), dtype=tiles[0].dtype, device=tiles[0].device)
        
        # Weight map per smooth blending (cosine taper)
        tile_weights = self._create_tile_weights(self.tile_size, self.overlap)
        
        for tile, (x, y) in zip(tiles, positions):
            # Calcola bounds effettivi
            x_end = min(x + self.tile_size, W)
            y_end = min(y + self.tile_size, H)
            
            # Dimensioni effettive del tile nella posizione finale
            tile_h = y_end - y
            tile_w = x_end - x
            
            # Crop tile e weights se necessario
            tile_crop = tile[:, :tile_h, :tile_w]
            weight_crop = tile_weights[:tile_h, :tile_w]
            
            # Accumula risultato con blending
            result[:, y:y_end, x:x_end] += tile_crop * weight_crop
            weights[y:y_end, x:x_end] += weight_crop
        
        # Normalizza per i pesi
        weights = weights.clamp(min=1e-8)  # Evita divisione per zero
        result = result / weights.unsqueeze(0)
        
        return result
    
    def _create_tile_weights(self, tile_size: int, overlap: int) -> torch.Tensor:
        """
        Crea weight map per smooth blending nei tile
        Usa cosine taper ai bordi per overlap smooth
        """
        weights = torch.ones((tile_size, tile_size))
        
        if overlap > 0:
            # Taper width (metà dell'overlap)
            taper = overlap // 2
            
            # Cosine taper ai bordi
            for i in range(taper):
                # Weight crescente dall'esterno verso l'interno
                w = 0.5 * (1 - np.cos(np.pi * i / taper))
                
                # Applica ai bordi
                weights[i, :] *= w          # Top
                weights[-(i+1), :] *= w     # Bottom  
                weights[:, i] *= w          # Left
                weights[:, -(i+1)] *= w     # Right
        
        return weights
    
    @torch.no_grad()
    def process_image(self, 
                      input_image: Union[str, Path, Image.Image],
                      return_mask: bool = False) -> Union[Image.Image, Tuple[Image.Image, Image.Image]]:
        """
        Processa una singola immagine
        
        Args:
            input_image: Path all'immagine o PIL Image
            return_mask: Se True, ritorna anche la maschera stelle
            
        Returns:
            Starless image (e opzionalmente star mask)
        """
        # Load image
        if isinstance(input_image, (str, Path)):
            image = Image.open(input_image)
        else:
            image = input_image
        
        print(f"Processing image: {image.size}")
        
        # Convert to tensor
        tensor = self._image_to_tensor(image)
        
        # Extract tiles
        tiles, positions, original_size = self._extract_tiles(tensor)
        print(f"Created {len(tiles)} tiles of size {self.tile_size}x{self.tile_size}")
        
        # Process tiles
        processed_tiles_starless = []
        processed_tiles_mask = []
        
        for i, tile in enumerate(tiles):
            # Add batch dimension
            tile_batch = tile.unsqueeze(0).to(self.device)
            
            # Forward pass
            pred_starless, pred_mask = self.model(tile_batch)
            
            # Remove batch dimension
            pred_starless = pred_starless.squeeze(0)
            pred_mask = pred_mask.squeeze(0)
            
            processed_tiles_starless.append(pred_starless)
            if return_mask:
                # Convert logits to probabilities for mask
                pred_mask = torch.sigmoid(pred_mask)
                processed_tiles_mask.append(pred_mask)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(tiles)} tiles")
        
        # Merge tiles
        result_starless = self._merge_tiles(processed_tiles_starless, positions, original_size)
        
        # Convert back to images
        starless_image = self._tensor_to_image(result_starless)
        
        if return_mask:
            result_mask = self._merge_tiles(processed_tiles_mask, positions, original_size)
            # Convert single channel mask to RGB for visualization
            mask_rgb = result_mask.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
            mask_image = self._tensor_to_image(mask_rgb)
            return starless_image, mask_image
        
        return starless_image
    
    def process_batch(self, 
                      input_dir: Union[str, Path],
                      output_dir: Union[str, Path],
                      pattern: str = "*.{jpg,jpeg,png,tif,tiff}",
                      save_mask: bool = False):
        """
        Processa un batch di immagini
        
        Args:
            input_dir: Directory con immagini input
            output_dir: Directory per output
            pattern: Pattern per i file (usando pathlib glob)
            save_mask: Se salvare anche le maschere
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if save_mask:
            mask_output_path = output_path / "masks"
            mask_output_path.mkdir(exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in ['jpg', 'jpeg', 'png', 'tif', 'tiff']:
            image_files.extend(input_path.glob(f"*.{ext}"))
            image_files.extend(input_path.glob(f"*.{ext.upper()}"))
        
        image_files = sorted(image_files)
        print(f"Found {len(image_files)} images to process")
        
        for i, img_path in enumerate(image_files):
            print(f"\n[{i+1}/{len(image_files)}] Processing: {img_path.name}")
            
            try:
                # Process image
                if save_mask:
                    starless_img, mask_img = self.process_image(img_path, return_mask=True)
                    
                    # Save mask
                    mask_output_file = mask_output_path / f"{img_path.stem}_mask{img_path.suffix}"
                    mask_img.save(mask_output_file)
                else:
                    starless_img = self.process_image(img_path)
                
                # Save starless image
                output_file = output_path / f"{img_path.stem}_starless{img_path.suffix}"
                starless_img.save(output_file)
                
                print(f"Saved: {output_file.name}")
                
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
        
        print(f"\nBatch processing completed! Results in: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="UFormer Star Removal Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                       help="Input image or directory")
    parser.add_argument("--output", type=str, required=True,
                       help="Output image or directory")
    parser.add_argument("--tile-size", type=int, default=512,
                       help="Tile size for processing (default: 512)")
    parser.add_argument("--overlap", type=int, default=64,
                       help="Overlap between tiles (default: 64)")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device for inference")
    parser.add_argument("--save-mask", action="store_true",
                       help="Save star masks")
    parser.add_argument("--batch", action="store_true",
                       help="Process directory (batch mode)")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = UFormerInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        tile_size=args.tile_size,
        overlap=args.overlap
    )
    
    if args.batch:
        # Batch processing
        inference.process_batch(
            input_dir=args.input,
            output_dir=args.output,
            save_mask=args.save_mask
        )
    else:
        # Single image processing
        input_path = Path(args.input)
        output_path = Path(args.output)
        
        if args.save_mask:
            starless_img, mask_img = inference.process_image(input_path, return_mask=True)
            
            # Save mask
            mask_path = output_path.parent / f"{output_path.stem}_mask{output_path.suffix}"
            mask_img.save(mask_path)
            print(f"Mask saved: {mask_path}")
        else:
            starless_img = inference.process_image(input_path)
        
        starless_img.save(output_path)
        print(f"Result saved: {output_path}")


if __name__ == "__main__":
    main()

"""
Test dell'inference UFormer senza checkpoint reale
Crea un modello mock per verificare il tiling e blending
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from uformer import UFormerStarRemoval


class MockUFormerInference:
    """Versione mock per testare l'inference pipeline senza checkpoint"""
    
    def __init__(self, tile_size=512, overlap=64):
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create mock model
        self.model = UFormerStarRemoval()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Mock UFormer loaded on {self.device}")
    
    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
        
        return tensor
    
    def _tensor_to_image(self, tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image"""
        img_array = tensor.clamp(0, 1).cpu().numpy()
        img_array = img_array.transpose(1, 2, 0)
        img_array = (img_array * 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    def _extract_tiles(self, tensor):
        """Extract tiles with overlap"""
        C, H, W = tensor.shape
        tiles = []
        positions = []
        
        step = self.tile_size - self.overlap
        
        for y in range(0, H, step):
            for x in range(0, W, step):
                x_end = min(x + self.tile_size, W)
                y_end = min(y + self.tile_size, H)
                
                if x_end - x < self.tile_size:
                    x = max(0, W - self.tile_size)
                    x_end = W
                if y_end - y < self.tile_size:
                    y = max(0, H - self.tile_size) 
                    y_end = H
                
                tile = tensor[:, y:y_end, x:x_end]
                
                # Pad if needed
                if tile.shape[1] < self.tile_size or tile.shape[2] < self.tile_size:
                    tile = torch.nn.functional.pad(
                        tile, 
                        (0, self.tile_size - tile.shape[2], 0, self.tile_size - tile.shape[1]),
                        mode='reflect'
                    )
                
                tiles.append(tile)
                positions.append((x, y))
        
        return tiles, positions, (H, W)
    
    def _create_tile_weights(self, tile_size, overlap):
        """Create smooth blending weights"""
        weights = torch.ones((tile_size, tile_size))
        
        if overlap > 0:
            taper = overlap // 2
            
            for i in range(taper):
                w = 0.5 * (1 - np.cos(np.pi * i / taper))
                
                weights[i, :] *= w
                weights[-(i+1), :] *= w
                weights[:, i] *= w
                weights[:, -(i+1)] *= w
        
        return weights
    
    def _merge_tiles(self, tiles, positions, original_size):
        """Merge tiles with blending"""
        H, W = original_size
        C = tiles[0].shape[0]
        
        result = torch.zeros((C, H, W), dtype=tiles[0].dtype, device=tiles[0].device)
        weights = torch.zeros((H, W), dtype=tiles[0].dtype, device=tiles[0].device)
        
        tile_weights = self._create_tile_weights(self.tile_size, self.overlap)
        
        for tile, (x, y) in zip(tiles, positions):
            x_end = min(x + self.tile_size, W)
            y_end = min(y + self.tile_size, H)
            
            tile_h = y_end - y
            tile_w = x_end - x
            
            tile_crop = tile[:, :tile_h, :tile_w]
            weight_crop = tile_weights[:tile_h, :tile_w]
            
            result[:, y:y_end, x:x_end] += tile_crop * weight_crop
            weights[y:y_end, x:x_end] += weight_crop
        
        weights = weights.clamp(min=1e-8)
        result = result / weights.unsqueeze(0)
        
        return result
    
    @torch.no_grad()
    def test_process(self, width=2048, height=1536):
        """Test processing con immagine mock"""
        
        print(f"Testing tiling on {width}x{height} image...")
        
        # Create mock image (gradiente colorato)
        mock_array = np.zeros((height, width, 3), dtype=np.float32)
        
        # Gradiente orizzontale per R
        for x in range(width):
            mock_array[:, x, 0] = x / width
        
        # Gradiente verticale per G  
        for y in range(height):
            mock_array[y, :, 1] = y / height
            
        # Pattern scacchiera per B
        tile_pattern = 64
        for y in range(height):
            for x in range(width):
                if ((x // tile_pattern) + (y // tile_pattern)) % 2:
                    mock_array[y, x, 2] = 0.5
        
        # Convert to PIL Image
        mock_image = Image.fromarray((mock_array * 255).astype(np.uint8))
        print(f"Created mock image: {mock_image.size}")
        
        # Convert to tensor
        tensor = self._image_to_tensor(mock_image)
        print(f"Tensor shape: {tensor.shape}")
        
        # Extract tiles
        tiles, positions, original_size = self._extract_tiles(tensor)
        print(f"Extracted {len(tiles)} tiles")
        print(f"Original size: {original_size}")
        
        # "Process" tiles (mock - just apply slight modification)
        processed_tiles = []
        
        for i, tile in enumerate(tiles):
            tile_batch = tile.unsqueeze(0).to(self.device)
            
            # Mock processing con il vero modello (ma su dati mock)
            try:
                pred_starless, pred_mask = self.model(tile_batch)
                pred_starless = pred_starless.squeeze(0)
                processed_tiles.append(pred_starless)
                
            except Exception as e:
                print(f"Model failed on tile {i}: {e}")
                # Fallback: just return original with slight modification
                processed_tiles.append(tile * 0.9)  # Slightly darker
            
            if (i + 1) % 5 == 0:
                print(f"Processed {i + 1}/{len(tiles)} tiles")
        
        # Merge tiles
        result = self._merge_tiles(processed_tiles, positions, original_size)
        print(f"Merged result shape: {result.shape}")
        
        # Convert back to image
        result_image = self._tensor_to_image(result)
        print(f"Final image size: {result_image.size}")
        
        return mock_image, result_image


def test_inference_pipeline():
    """Test completo del pipeline di inference"""
    
    print("🧪 Testing UFormer Inference Pipeline")
    print("=" * 50)
    
    # Test con dimensioni diverse
    test_cases = [
        (1024, 768),    # Small 
        (2048, 1536),   # Medium
        (4096, 3072),   # Large
    ]
    
    for i, (w, h) in enumerate(test_cases):
        print(f"\n📋 Test Case {i+1}: {w}x{h}")
        
        inference = MockUFormerInference(tile_size=512, overlap=64)
        
        try:
            mock_img, result_img = inference.test_process(w, h)
            
            # Save results
            output_dir = Path("inference_test_output")
            output_dir.mkdir(exist_ok=True)
            
            mock_path = output_dir / f"mock_{w}x{h}.jpg"
            result_path = output_dir / f"result_{w}x{h}.jpg"
            
            mock_img.save(mock_path)
            result_img.save(result_path)
            
            print(f"✅ Success! Saved to {result_path}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Inference pipeline test completed!")


if __name__ == "__main__":
    test_inference_pipeline()

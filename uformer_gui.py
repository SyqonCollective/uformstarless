"""
GUI per testare checkpoint UFormer con tiling ottimizzato per Mac M1 Metal
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
import numpy as np
from pathlib import Path
import yaml
import threading
import time

from uformer import UFormerStarRemoval


class UFormerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UFormer Star Removal - Checkpoint Tester")
        self.root.geometry("800x600")
        
        # Model state
        self.model = None
        self.device = self.get_device()
        self.checkpoint_path = None
        self.image_path = None
        self.tile_size = 512
        self.overlap = 64  # Optimized overlap
        
        self.setup_ui()
        self.update_status(f"Ready - Device: {self.device}")
        
    def get_device(self):
        """Get best available device for Mac M1"""
        if torch.backends.mps.is_available():
            return torch.device("mps")  # Metal Performance Shaders
        else:
            return torch.device("cpu")
    
    def setup_ui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model loading section
        model_frame = ttk.LabelFrame(main_frame, text="Model Loading", padding="10")
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.checkpoint_label = ttk.Label(model_frame, text="No checkpoint loaded")
        self.checkpoint_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        ttk.Button(model_frame, text="Load Checkpoint", 
                  command=self.load_checkpoint).grid(row=0, column=1, sticky=tk.E)
        
        # Image loading section
        image_frame = ttk.LabelFrame(main_frame, text="Image Processing", padding="10")
        image_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="No image loaded")
        self.image_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        ttk.Button(image_frame, text="Load Image", 
                  command=self.load_image).grid(row=0, column=1, padx=(0, 10))
        
        # Store reference to process button for easy access
        self.process_button = ttk.Button(image_frame, text="Process Image", 
                                        command=self.process_image, state='disabled')
        self.process_button.grid(row=0, column=2)
        
        # Processing settings
        settings_frame = ttk.LabelFrame(main_frame, text="Processing Settings", padding="10")
        settings_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(settings_frame, text="Tile Size:").grid(row=0, column=0, sticky=tk.W)
        self.tile_var = tk.IntVar(value=512)
        tile_combo = ttk.Combobox(settings_frame, textvariable=self.tile_var, 
                                 values=[256, 512, 1024], state='readonly', width=10)
        tile_combo.grid(row=0, column=1, padx=(5, 20))
        
        ttk.Label(settings_frame, text="Overlap:").grid(row=0, column=2, sticky=tk.W)
        self.overlap_var = tk.IntVar(value=64)
        overlap_combo = ttk.Combobox(settings_frame, textvariable=self.overlap_var,
                                   values=[32, 64, 128], state='readonly', width=10)
        overlap_combo.grid(row=0, column=3, padx=(5, 20))
        
        # No tiling checkbox
        self.no_tiling_var = tk.BooleanVar(value=False)
        no_tiling_check = ttk.Checkbutton(settings_frame, text="No Tiling", 
                                         variable=self.no_tiling_var,
                                         command=self.toggle_tiling_settings)
        no_tiling_check.grid(row=0, column=4, padx=(20, 0))
        
        # Store widgets for enable/disable
        self.tiling_widgets = [tile_combo, overlap_combo]
        
        # Preview area
        preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding="10")
        preview_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.preview_canvas = tk.Canvas(preview_frame, width=400, height=300, bg='gray90')
        self.preview_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress and status
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=5, column=0, columnspan=2, sticky=tk.W)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def toggle_tiling_settings(self):
        """Enable/disable tiling settings based on checkbox"""
        if self.no_tiling_var.get():
            # Disable tiling controls
            for widget in self.tiling_widgets:
                widget.config(state='disabled')
        else:
            # Enable tiling controls
            for widget in self.tiling_widgets:
                widget.config(state='readonly')
    
    def load_checkpoint(self):
        """Load UFormer checkpoint"""
        file_path = filedialog.askopenfilename(
            title="Select UFormer Checkpoint",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")],
            initialdir="experiments"
        )
        
        if not file_path:
            return
            
        try:
            self.update_status("Loading checkpoint...")
            
            # Initialize model (use same config as training)
            self.model = UFormerStarRemoval(
                embed_dim=96,
                window_size=8,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24]
            )
            
            # Load checkpoint
            checkpoint = torch.load(file_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                loss = checkpoint.get('val_loss', checkpoint.get('loss', 'unknown'))
                self.checkpoint_label.config(text=f"Loaded: {Path(file_path).name} (Epoch: {epoch}, Loss: {loss:.4f})" if isinstance(loss, (int, float)) else f"Loaded: {Path(file_path).name}")
            else:
                self.model.load_state_dict(checkpoint)
                self.checkpoint_label.config(text=f"Loaded: {Path(file_path).name}")
            
            self.model.to(self.device)
            self.model.eval()
            self.checkpoint_path = file_path
            
            # Enable process button if image is loaded
            if self.image_path:
                self.enable_processing()
            
            self.update_status(f"Checkpoint loaded successfully on {self.device}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load checkpoint: {str(e)}")
            self.update_status("Failed to load checkpoint")
    
    def load_image(self):
        """Load image for processing"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            self.image_path = file_path
            
            # Load and show preview
            image = Image.open(file_path).convert('RGB')
            self.original_image = image
            
            # Create preview
            preview_size = (400, 300)
            preview_img = image.copy()
            preview_img.thumbnail(preview_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            self.preview_photo = ImageTk.PhotoImage(preview_img)
            
            # Clear canvas and show image
            self.preview_canvas.delete("all")
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # Canvas is initialized
                x = (canvas_width - preview_img.width) // 2
                y = (canvas_height - preview_img.height) // 2
                self.preview_canvas.create_image(x, y, anchor=tk.NW, image=self.preview_photo)
            
            self.image_label.config(text=f"Loaded: {Path(file_path).name} ({image.width}x{image.height})")
            
            # Enable process button if model is loaded
            if self.model:
                self.enable_processing()
            
            self.update_status(f"Image loaded: {image.width}x{image.height}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.update_status("Failed to load image")
    
    def enable_processing(self):
        """Enable the process button"""
        self.process_button.config(state='normal')
    
    def process_image(self):
        """Process the loaded image with tiling"""
        if not self.model or not self.image_path:
            messagebox.showwarning("Warning", "Please load both checkpoint and image first")
            return
        
        # Start processing in separate thread
        threading.Thread(target=self._process_image_thread, daemon=True).start()
    
    def _process_image_thread(self):
        """Process image in separate thread"""
        try:
            self.progress.start()
            self.update_status("Processing image...")
            
            # Get current settings
            no_tiling = self.no_tiling_var.get()
            tile_size = self.tile_var.get()
            overlap = self.overlap_var.get()
            
            # Load image
            image = Image.open(self.image_path).convert('RGB')
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # Process based on tiling setting
            if no_tiling:
                result_np = self.process_full_image(image_np)
            else:
                result_np = self.process_with_tiling(image_np, tile_size, overlap)
            
            # Convert back to PIL and save
            result_img = Image.fromarray((result_np * 255).astype(np.uint8))
            
            # Save in same directory as input
            input_path = Path(self.image_path)
            suffix = "_starless_notiling" if no_tiling else "_starless"
            output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
            result_img.save(output_path)
            
            self.progress.stop()
            self.update_status(f"Processed and saved: {output_path.name}")
            
            # Show result dialog
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                f"Image processed successfully!\nSaved as: {output_path.name}"))
            
        except Exception as e:
            self.progress.stop()
            self.update_status("Processing failed")
            error_msg = str(e)  # Capture the error message
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Processing failed: {msg}"))
    
    def process_full_image(self, image_np):
        """Process image without tiling - with maximum compatibility"""
        h, w, c = image_np.shape
        original_h, original_w = h, w
        self.update_status(f"Processing full image ({w}x{h}) without tiling...")
        
        try:
            with torch.no_grad():
                self.update_status("Finding UFormer-compatible dimensions...")
                
                # UFormer is very picky about dimensions. Use training-compatible sizes.
                # The model was trained on 512x512, so use sizes that work well
                
                # Find the best size that's <= original and works with UFormer
                # Common working sizes: multiples of 64 work better than just 32
                def find_safe_size(size):
                    # Use multiples of 64 for maximum compatibility  
                    # If size is very large, cap it for memory/performance
                    max_safe_size = 2048  # Conservative limit for no-tiling
                    target = min(size, max_safe_size)
                    target = (target // 64) * 64
                    if target == 0:
                        target = 64
                    return target
                
                target_w = find_safe_size(w)
                target_h = find_safe_size(h)
                
                # Additional safety: if dimensions are still problematic, use 512x512 equivalent aspect
                if target_w > 1024 or target_h > 1024:
                    self.update_status("Using conservative resize for large image...")
                    # Maintain aspect ratio but cap at reasonable size
                    aspect_ratio = w / h
                    if aspect_ratio > 1:
                        # Landscape
                        target_w = 1024
                        target_h = int(1024 / aspect_ratio)
                        target_h = (target_h // 64) * 64
                    else:
                        # Portrait or square
                        target_h = 1024  
                        target_w = int(1024 * aspect_ratio)
                        target_w = (target_w // 64) * 64
                
                # Resize if needed
                if target_w != w or target_h != h:
                    resize_factor = min(target_w/w, target_h/h) * 100
                    self.update_status(f"Smart resize: {w}x{h} → {target_w}x{target_h} ({resize_factor:.1f}%)")
                    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                    image_resized = image_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    image_np = np.array(image_resized).astype(np.float32) / 255.0
                    h, w = target_h, target_w
                
                # Convert to tensor
                image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                
                # Memory check
                tensor_size_mb = image_tensor.element_size() * image_tensor.nelement() / (1024 * 1024)
                self.update_status(f"Processing {w}x{h} tensor ({tensor_size_mb:.1f}MB)...")
                
                # Test if dimensions work with a small forward pass first
                try:
                    # Small test to validate dimensions
                    test_tensor = torch.zeros(1, 3, 64, 64).to(self.device)
                    _, _ = self.model(test_tensor)
                    del test_tensor
                    
                    # Now process the real image
                    output_tensor, _ = self.model(image_tensor)
                    result_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    
                except Exception as model_error:
                    # If even the safe size fails, fall back to 512x512
                    self.update_status("Using fail-safe 512x512 processing...")
                    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
                    image_512 = image_pil.resize((512, 512), Image.Resampling.LANCZOS)
                    image_512_np = np.array(image_512).astype(np.float32) / 255.0
                    
                    tensor_512 = torch.from_numpy(image_512_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                    output_tensor, _ = self.model(tensor_512)
                    result_512 = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    
                    # Resize 512 result back to target size
                    result_pil = Image.fromarray((result_512 * 255).astype(np.uint8))
                    result_resized = result_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    result_np = np.array(result_resized).astype(np.float32) / 255.0
                
                # Resize back to original size if needed
                if target_w != original_w or target_h != original_h:
                    self.update_status(f"Restoring original size: {w}x{h} → {original_w}x{original_h}")
                    result_pil = Image.fromarray((result_np * 255).astype(np.uint8))
                    result_final = result_pil.resize((original_w, original_h), Image.Resampling.LANCZOS) 
                    result_np = np.array(result_final).astype(np.float32) / 255.0
                
                self.update_status("No-tiling processing complete!")
                
            return np.clip(result_np, 0, 1)
            
        except torch.cuda.OutOfMemoryError:
            tensor_size_mb = tensor_size_mb if 'tensor_size_mb' in locals() else 0
            raise Exception(f"GPU out of memory for {original_w}x{original_h} image ({tensor_size_mb:.1f}MB). Try using tiling instead.")
        except Exception as e:
            raise Exception(f"No-tiling failed even with safe dimensions. Use tiling for this image. Error: {str(e)}")
    
    def process_with_tiling(self, image_np, tile_size, overlap):
        """Process large image with optimized tiling"""
        h, w, c = image_np.shape
        
        # Calculate tiling parameters
        stride = tile_size - overlap
        n_tiles_h = (h - overlap + stride - 1) // stride
        n_tiles_w = (w - overlap + stride - 1) // stride
        
        # Output array
        output = np.zeros_like(image_np)
        weight_map = np.zeros((h, w, 1))
        
        total_tiles = n_tiles_h * n_tiles_w
        processed_tiles = 0
        
        with torch.no_grad():
            for i in range(n_tiles_h):
                for j in range(n_tiles_w):
                    # Calculate tile boundaries
                    y1 = i * stride
                    x1 = j * stride
                    y2 = min(y1 + tile_size, h)
                    x2 = min(x1 + tile_size, w)
                    
                    # Extract tile
                    tile = image_np[y1:y2, x1:x2]
                    
                    # Pad if necessary
                    pad_h = tile_size - tile.shape[0]
                    pad_w = tile_size - tile.shape[1]
                    if pad_h > 0 or pad_w > 0:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                    
                    # Convert to tensor
                    tile_tensor = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                    
                    # Process with model
                    output_tensor, _ = self.model(tile_tensor)
                    processed_tile = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    
                    # Remove padding
                    if pad_h > 0 or pad_w > 0:
                        processed_tile = processed_tile[:tile_size-pad_h, :tile_size-pad_w]
                    
                    # Add to output with weighting for overlap blending
                    actual_h, actual_w = processed_tile.shape[:2]
                    
                    # Create weight mask for smooth blending
                    weight_tile = np.ones((actual_h, actual_w, 1))
                    
                    # Feather edges for smooth blending
                    feather = min(overlap // 2, 32)  # Increased from 16 to 32 for smoother blending
                    if feather > 0:
                        # Top edge
                        if i > 0:
                            weight_tile[:feather, :, :] *= np.linspace(0, 1, feather).reshape(-1, 1, 1)
                        # Bottom edge  
                        if i < n_tiles_h - 1:
                            weight_tile[-feather:, :, :] *= np.linspace(1, 0, feather).reshape(-1, 1, 1)
                        # Left edge
                        if j > 0:
                            weight_tile[:, :feather, :] *= np.linspace(0, 1, feather).reshape(1, -1, 1)
                        # Right edge
                        if j < n_tiles_w - 1:
                            weight_tile[:, -feather:, :] *= np.linspace(1, 0, feather).reshape(1, -1, 1)
                    
                    # Add weighted tile to output
                    output[y1:y1+actual_h, x1:x1+actual_w] += processed_tile * weight_tile
                    weight_map[y1:y1+actual_h, x1:x1+actual_w] += weight_tile
                    
                    processed_tiles += 1
                    
                    # Update progress (on main thread)
                    progress_pct = (processed_tiles / total_tiles) * 100
                    self.root.after(0, lambda p=progress_pct, pt=processed_tiles, tt=total_tiles: 
                        self.update_status(f"Processing tiles... {p:.1f}% ({pt}/{tt})"))
        
        # Normalize by weights
        weight_map = np.maximum(weight_map, 1e-6)  # Avoid division by zero
        output = output / weight_map
        
        return np.clip(output, 0, 1)


def main():
    root = tk.Tk()
    app = UFormerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

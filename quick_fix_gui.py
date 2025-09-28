"""
GUI per UFormer Quick Fix - NO QUADRETTI!
Versione adattata per il modello con Shifted Windows
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

# Import Quick Fix UFormer
from quick_fix_uformer import quick_fix_uformer
from uformer import UFormerStarRemoval


class QuickFixUFormerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UFormer Quick Fix - NO QUADRETTI! 🚀")
        self.root.geometry("900x700")
        
        # Model state
        self.model = None
        self.device = self.get_device()
        self.checkpoint_path = None
        self.image_path = None
        self.tile_size = 512
        self.overlap = 64
        self.use_quick_fix = True  # Default: usa versione Quick Fix
        
        self.setup_ui()
        self.update_status(f"Ready - Device: {self.device} - Quick Fix Mode: ENABLED ✨")
        
    def get_device(self):
        """Get best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")  # Mac M1/M2
        else:
            return torch.device("cpu")
    
    def setup_ui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Model loading section
        model_frame = ttk.LabelFrame(main_frame, text="🎯 Model Loading (Quick Fix Mode)", padding="10")
        model_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Quick Fix toggle
        self.quick_fix_var = tk.BooleanVar(value=True)
        quick_fix_check = ttk.Checkbutton(
            model_frame, 
            text="✨ Enable Quick Fix (Eliminates quadretti with Shifted Windows)",
            variable=self.quick_fix_var,
            command=self.toggle_quick_fix
        )
        quick_fix_check.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Checkpoint status
        self.checkpoint_label = ttk.Label(model_frame, text="No model loaded")
        self.checkpoint_label.grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        
        ttk.Button(model_frame, text="Load UFormer Checkpoint", 
                  command=self.load_checkpoint).grid(row=1, column=1, sticky=tk.E)
        
        # Quick buttons for common paths
        common_frame = ttk.Frame(model_frame)
        common_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        ttk.Button(common_frame, text="Load: best_model.pth", 
                  command=self.load_best_model).grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(common_frame, text="Browse Quick Fix Model", 
                  command=self.browse_quick_fix_model).grid(row=0, column=1, padx=(5, 0))
        
        # Image loading section
        image_frame = ttk.LabelFrame(main_frame, text="📷 Image Loading", padding="10")
        image_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.image_label = ttk.Label(image_frame, text="No image loaded")
        self.image_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        
        ttk.Button(image_frame, text="Load Image", 
                  command=self.load_image).grid(row=0, column=1, sticky=tk.E)
        
        # Processing section
        process_frame = ttk.LabelFrame(main_frame, text="🚀 Processing", padding="10")
        process_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Tile size
        ttk.Label(process_frame, text="Tile Size:").grid(row=0, column=0, sticky=tk.W)
        self.tile_var = tk.IntVar(value=512)
        tile_combo = ttk.Combobox(process_frame, textvariable=self.tile_var, 
                                 values=[256, 512, 768, 1024], width=10, state="readonly")
        tile_combo.grid(row=0, column=1, sticky=tk.W, padx=(5, 20))
        
        # Overlap
        ttk.Label(process_frame, text="Overlap:").grid(row=0, column=2, sticky=tk.W)
        self.overlap_var = tk.IntVar(value=64)
        overlap_combo = ttk.Combobox(process_frame, textvariable=self.overlap_var,
                                   values=[32, 64, 128], width=10, state="readonly")
        overlap_combo.grid(row=0, column=3, sticky=tk.W, padx=(5, 0))
        
        # Process button
        self.process_btn = ttk.Button(process_frame, text="🎯 Process Image (NO QUADRETTI!)", 
                                    command=self.process_image, state="disabled")
        self.process_btn.grid(row=1, column=0, columnspan=4, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="📊 Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Image display frame
        self.canvas_frame = ttk.Frame(results_frame)
        self.canvas_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=5, column=0, columnspan=2, sticky=tk.W)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
    
    def toggle_quick_fix(self):
        """Toggle Quick Fix mode"""
        self.use_quick_fix = self.quick_fix_var.get()
        if self.use_quick_fix:
            self.update_status("✨ Quick Fix ENABLED - Will eliminate quadretti!")
            self.process_btn.config(text="🎯 Process Image (NO QUADRETTI!)")
        else:
            self.update_status("⚠️  Quick Fix DISABLED - May have quadretti artifacts")
            self.process_btn.config(text="Process Image (Standard)")
    
    def load_checkpoint(self):
        """Load UFormer checkpoint"""
        file_path = filedialog.askopenfilename(
            title="Select UFormer Checkpoint",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")],
            initialdir="."
        )
        
        if not file_path:
            return
            
        self._load_model(file_path)
    
    def load_best_model(self):
        """Quick load best_model.pth"""
        checkpoint_path = "experiments/uformer_debug/checkpoints/best_model.pth"
        if Path(checkpoint_path).exists():
            self._load_model(checkpoint_path)
        else:
            messagebox.showerror("Error", f"File not found: {checkpoint_path}")
    
    def browse_quick_fix_model(self):
        """Browse and load a Quick Fix model"""
        file_path = filedialog.askopenfilename(
            title="Select Quick Fix Model",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")],
            initialdir="."
        )
        
        if file_path:
            self._load_model(file_path, is_quick_fix=True)
    
    def _load_model(self, file_path, is_quick_fix=False):
        """Internal method to load model"""
        try:
            self.update_status("Loading model...")
            self.progress.start()
            
            if is_quick_fix:
                # Load pre-generated Quick Fix model using the quick_fix_uformer function
                self.model = quick_fix_uformer(file_path, device=str(self.device))
                
                self.checkpoint_label.config(text=f"✨ Loaded Quick Fix: {Path(file_path).name}")
                self.quick_fix_var.set(False)  # Already fixed model
                self.use_quick_fix = False
                
            elif self.use_quick_fix:
                # Apply Quick Fix to original checkpoint
                self.model = quick_fix_uformer(file_path, device=str(self.device))
                self.checkpoint_label.config(text=f"✨ Quick Fixed: {Path(file_path).name}")
                
            else:
                # Standard UFormer loading
                self.model = UFormerStarRemoval(
                    embed_dim=96,
                    window_size=8, 
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24]
                )
                
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
            
            mode = "Quick Fix" if (self.use_quick_fix or is_quick_fix) else "Standard"
            self.update_status(f"✅ Model loaded successfully on {self.device} - Mode: {mode}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load checkpoint: {str(e)}")
            self.update_status("❌ Failed to load model")
        finally:
            self.progress.stop()
    
    def load_image(self):
        """Load input image"""
        file_path = filedialog.askopenfilename(
            title="Select Input Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tiff *.bmp"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            # Load and display image info
            image = Image.open(file_path)
            self.image_path = file_path
            self.image_label.config(text=f"Loaded: {Path(file_path).name} ({image.size[0]}x{image.size[1]})")
            
            # Enable process button if model is loaded
            if self.model:
                self.enable_processing()
                
            self.update_status(f"✅ Image loaded: {image.size[0]}x{image.size[1]}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def enable_processing(self):
        """Enable the process button"""
        self.process_btn.config(state="normal")
    
    def process_image(self):
        """Process the loaded image"""
        if not self.model or not self.image_path:
            messagebox.showerror("Error", "Please load both model and image first")
            return
        
        # Run processing in separate thread to avoid GUI freeze
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        """Process image in separate thread"""
        try:
            self.root.after(0, lambda: self.progress.start())
            
            # Get settings
            tile_size = self.tile_var.get()
            overlap = self.overlap_var.get()
            
            mode = "Quick Fix (NO quadretti)" if (self.use_quick_fix or "no_quadretti" in str(self.checkpoint_path)) else "Standard"
            self.root.after(0, lambda: self.update_status(f"🚀 Processing with {mode}... Tile: {tile_size}, Overlap: {overlap}"))
            
            # Load and preprocess image
            image = Image.open(self.image_path).convert('RGB')
            input_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(self.device)
            
            # Process with tiling
            with torch.no_grad():
                starless_output, mask_output = self.tile_process(input_tensor, tile_size, overlap)
            
            # Convert results to images
            starless_img = self.tensor_to_pil(starless_output)
            mask_img = self.tensor_to_pil(mask_output, is_mask=True)
            
            # Save results
            input_path = Path(self.image_path)
            suffix = "_quick_fix" if (self.use_quick_fix or "no_quadretti" in str(self.checkpoint_path)) else "_standard"
            
            starless_path = input_path.parent / f"{input_path.stem}{suffix}_starless.png"
            mask_path = input_path.parent / f"{input_path.stem}{suffix}_mask.png"
            
            starless_img.save(starless_path)
            mask_img.save(mask_path)
            
            # Display results
            self.root.after(0, lambda: self.display_results(starless_img, mask_img))
            self.root.after(0, lambda: self.update_status(f"✅ Processing complete! Saved: {starless_path.name}, {mask_path.name}"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            self.root.after(0, lambda: self.update_status("❌ Processing failed"))
        finally:
            self.root.after(0, lambda: self.progress.stop())
    
    def tile_process(self, input_tensor, tile_size, overlap):
        """Process image with tiling"""
        B, C, H, W = input_tensor.shape
        
        if H <= tile_size and W <= tile_size:
            # No tiling needed
            return self.model(input_tensor)
        
        # Calculate tiles
        stride = tile_size - overlap
        h_tiles = (H - overlap) // stride + (1 if (H - overlap) % stride else 0)
        w_tiles = (W - overlap) // stride + (1 if (W - overlap) % stride else 0)
        
        # Initialize output tensors
        starless_output = torch.zeros_like(input_tensor)
        mask_output = torch.zeros(B, 1, H, W, device=input_tensor.device)
        weight_map = torch.zeros(B, 1, H, W, device=input_tensor.device)
        
        # Process each tile
        for h in range(h_tiles):
            for w in range(w_tiles):
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
                tile_starless, tile_mask = self.model(tile)
                
                # Remove padding
                if pad_h > 0 or pad_w > 0:
                    tile_starless = tile_starless[:, :, :tile_size-pad_h, :tile_size-pad_w]
                    tile_mask = tile_mask[:, :, :tile_size-pad_h, :tile_size-pad_w]
                
                # Create weight for blending (higher weight in center)
                tile_h, tile_w = tile_starless.shape[2], tile_starless.shape[3]
                weight = torch.ones(1, 1, tile_h, tile_w, device=input_tensor.device)
                
                # Apply to output
                starless_output[:, :, h_start:h_end, w_start:w_end] += tile_starless * weight
                mask_output[:, :, h_start:h_end, w_start:w_end] += tile_mask * weight
                weight_map[:, :, h_start:h_end, w_start:w_end] += weight
        
        # Normalize by weight
        starless_output = starless_output / weight_map
        mask_output = mask_output / weight_map
        
        return starless_output, mask_output
    
    def tensor_to_pil(self, tensor, is_mask=False):
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
    
    def display_results(self, starless_img, mask_img):
        """Display processing results"""
        # Clear previous results
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # Resize images for display
        display_size = (300, 200)
        starless_display = starless_img.copy()
        starless_display.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        mask_display = mask_img.copy()
        mask_display.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        starless_photo = ImageTk.PhotoImage(starless_display)
        mask_photo = ImageTk.PhotoImage(mask_display)
        
        # Create display frame
        display_frame = ttk.Frame(self.canvas_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Starless result
        starless_frame = ttk.LabelFrame(display_frame, text="🌟 Starless Result", padding="5")
        starless_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        starless_label = ttk.Label(starless_frame, image=starless_photo)
        starless_label.image = starless_photo  # Keep reference
        starless_label.pack()
        
        # Mask result
        mask_frame = ttk.LabelFrame(display_frame, text="🎭 Star Mask", padding="5")
        mask_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        mask_label = ttk.Label(mask_frame, image=mask_photo)
        mask_label.image = mask_photo  # Keep reference
        mask_label.pack()
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update()


def main():
    root = tk.Tk()
    app = QuickFixUFormerGUI(root)
    
    # Configure window icon and style
    try:
        root.state('zoomed') if root.tk.call('tk', 'windowingsystem') == 'win32' else root.attributes('-zoomed', True)
    except:
        pass  # Fallback for different systems
    
    root.mainloop()


if __name__ == '__main__':
    main()

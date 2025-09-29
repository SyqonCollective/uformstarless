#!/usr/bin/env python3
"""
Starless GUI - Interfaccia per rimozione stelle con UFormer
Supporto GPU Metal M1 Pro con elaborazione tile 512x512 + overlap

Features:
- Caricamento checkpoint manuale
- Elaborazione con tiles 512x512 e overlap
- Accelerazione GPU Metal per M1 Pro
- Output nella stessa directory con suffisso "_starless"
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import threading
import queue
import math

# Aggiungi parent directory per imports
sys.path.append(str(Path(__file__).parent.parent))

from enhanced_uformer import EnhancedUFormerStarRemoval


class StarlessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Starless UFormer - Star Removal GUI")
        self.root.geometry("800x700")
        
        # Variabili
        self.model = None
        self.device = self.setup_device()
        self.checkpoint_path = tk.StringVar()
        self.image_path = tk.StringVar()
        self.progress_queue = queue.Queue()
        
        self.setup_ui()
        self.check_progress()
        
    def setup_device(self):
        """Setup device con supporto Metal per M1 Pro"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("🚀 Using Metal Performance Shaders (M1 Pro)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("🚀 Using CUDA GPU")
        else:
            device = torch.device("cpu")
            print("⚠️ Using CPU (slower)")
        return device
        
    def setup_ui(self):
        """Setup interfaccia utente"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Starless UFormer - Star Removal", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Device info
        device_label = ttk.Label(main_frame, text=f"Device: {self.device}", 
                                font=('Arial', 10))
        device_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
        
        # Checkpoint selection
        ttk.Label(main_frame, text="1. Select Model Checkpoint:", 
                 font=('Arial', 12, 'bold')).grid(row=2, column=0, sticky=tk.W, pady=(10, 5))
        
        checkpoint_frame = ttk.Frame(main_frame)
        checkpoint_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Entry(checkpoint_frame, textvariable=self.checkpoint_path, width=60).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(checkpoint_frame, text="Browse Checkpoint", 
                  command=self.browse_checkpoint).grid(row=0, column=1)
        ttk.Button(checkpoint_frame, text="Load Model", 
                  command=self.load_model).grid(row=0, column=2, padx=(10, 0))
        
        # Model status
        self.model_status = ttk.Label(main_frame, text="Model: Not loaded", 
                                     foreground="red")
        self.model_status.grid(row=4, column=0, columnspan=3, pady=(0, 15))
        
        # Image selection
        ttk.Label(main_frame, text="2. Select Image:", 
                 font=('Arial', 12, 'bold')).grid(row=5, column=0, sticky=tk.W, pady=(10, 5))
        
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Entry(image_frame, textvariable=self.image_path, width=60).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(image_frame, text="Browse Image", 
                  command=self.browse_image).grid(row=0, column=1)
        
        # Processing options
        ttk.Label(main_frame, text="3. Processing Options:", 
                 font=('Arial', 12, 'bold')).grid(row=7, column=0, sticky=tk.W, pady=(15, 5))
        
        options_frame = ttk.Frame(main_frame)
        options_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Label(options_frame, text="Tile Size:").grid(row=0, column=0, sticky=tk.W)
        self.tile_size = tk.IntVar(value=512)
        ttk.Entry(options_frame, textvariable=self.tile_size, width=10).grid(row=0, column=1, padx=(5, 15))
        
        ttk.Label(options_frame, text="Overlap:").grid(row=0, column=2, sticky=tk.W)
        self.overlap = tk.IntVar(value=64)
        ttk.Entry(options_frame, textvariable=self.overlap, width=10).grid(row=0, column=3, padx=(5, 0))
        
        # Process button
        self.process_btn = ttk.Button(main_frame, text="🚀 Start Star Removal", 
                                     command=self.start_processing, state="disabled")
        self.process_btn.grid(row=9, column=0, columnspan=3, pady=(10, 15))
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="blue")
        self.status_label.grid(row=11, column=0, columnspan=3)
        
        # Image preview
        ttk.Label(main_frame, text="Image Preview:", 
                 font=('Arial', 12, 'bold')).grid(row=12, column=0, sticky=tk.W, pady=(20, 5))
        
        self.image_label = ttk.Label(main_frame, text="No image selected")
        self.image_label.grid(row=13, column=0, columnspan=3, pady=(0, 10))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
    def browse_checkpoint(self):
        """Browse per checkpoint file"""
        filetypes = [
            ("PyTorch Models", "*.pth *.pt"),
            ("All Files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=filetypes
        )
        if filename:
            self.checkpoint_path.set(filename)
            
    def browse_image(self):
        """Browse per immagine"""
        filetypes = [
            ("Images", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp"),
            ("All Files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select Image",
            filetypes=filetypes
        )
        if filename:
            self.image_path.set(filename)
            self.show_image_preview(filename)
            self.update_process_button_state()
            
    def show_image_preview(self, image_path):
        """Mostra preview dell'immagine"""
        try:
            image = Image.open(image_path)
            # Resize per preview
            image.thumbnail((400, 300), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep reference
        except Exception as e:
            self.image_label.configure(text=f"Error loading image: {e}")
            
    def load_model(self):
        """Carica il modello dal checkpoint"""
        checkpoint_path = self.checkpoint_path.get()
        if not checkpoint_path or not Path(checkpoint_path).exists():
            messagebox.showerror("Error", "Please select a valid checkpoint file")
            return
            
        try:
            self.status_label.configure(text="Loading model...", foreground="orange")
            self.root.update()
            
            # Crea modello
            self.model = EnhancedUFormerStarRemoval(
                embed_dim=96,
                window_size=8,
                halo_size=4,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                focal_interval=999,  # Disable focal blocks
                shifted_window=True
            ).to(self.device)
            
            # Carica checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
            
            self.model_status.configure(text="Model: Loaded ✅", foreground="green")
            self.status_label.configure(text="Model loaded successfully", foreground="green")
            self.update_process_button_state()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.model_status.configure(text="Model: Error ❌", foreground="red")
            self.status_label.configure(text="Error loading model", foreground="red")
            
    def update_process_button_state(self):
        """Aggiorna stato del pulsante process"""
        if self.model is not None and self.image_path.get():
            self.process_btn.configure(state="normal")
        else:
            self.process_btn.configure(state="disabled")
            
    def start_processing(self):
        """Avvia processing in thread separato"""
        if not self.model or not self.image_path.get():
            return
            
        # Disabilita pulsante durante processing
        self.process_btn.configure(state="disabled")
        
        # Avvia thread
        thread = threading.Thread(target=self.process_image_thread)
        thread.daemon = True
        thread.start()
        
    def process_image_thread(self):
        """Processa immagine in thread separato"""
        try:
            image_path = self.image_path.get()
            tile_size = self.tile_size.get()
            overlap = self.overlap.get()
            
            # Update status
            self.progress_queue.put(("status", "Loading image...", "orange"))
            
            # Carica immagine
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Process with tiles
            self.progress_queue.put(("status", "Processing tiles...", "blue"))
            result = self.process_with_tiles(image_array, tile_size, overlap)
            
            # Save result
            self.progress_queue.put(("status", "Saving result...", "blue"))
            output_path = self.get_output_path(image_path)
            
            result_image = (result * 255).astype(np.uint8)
            Image.fromarray(result_image).save(output_path)
            
            self.progress_queue.put(("status", f"Completed! Saved to: {output_path}", "green"))
            self.progress_queue.put(("progress", 100))
            self.progress_queue.put(("enable_button", True))
            
        except Exception as e:
            self.progress_queue.put(("status", f"Error: {e}", "red"))
            self.progress_queue.put(("enable_button", True))
            
    def process_with_tiles(self, image_array, tile_size, overlap):
        """Processa immagine con tiles e overlap"""
        H, W, C = image_array.shape
        result = np.zeros_like(image_array)
        weight_map = np.zeros((H, W))
        
        # Calcola numero di tiles
        step = tile_size - overlap
        tiles_h = math.ceil(H / step)
        tiles_w = math.ceil(W / step)
        total_tiles = tiles_h * tiles_w
        
        processed_tiles = 0
        
        with torch.no_grad():
            for i in range(tiles_h):
                for j in range(tiles_w):
                    # Coordinate tile
                    start_h = i * step
                    end_h = min(start_h + tile_size, H)
                    start_w = j * step
                    end_w = min(start_w + tile_size, W)
                    
                    # Extract tile
                    tile = image_array[start_h:end_h, start_w:end_w]
                    
                    # Pad se necessario
                    pad_h = tile_size - tile.shape[0]
                    pad_w = tile_size - tile.shape[1]
                    if pad_h > 0 or pad_w > 0:
                        tile = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                    
                    # Convert to tensor
                    tile_tensor = torch.from_numpy(tile.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
                    
                    # Process
                    starless, _ = self.model(tile_tensor)
                    starless_np = starless.cpu().numpy()[0].transpose(1, 2, 0)
                    
                    # Remove padding
                    if pad_h > 0 or pad_w > 0:
                        starless_np = starless_np[:tile_size-pad_h, :tile_size-pad_w]
                    
                    # Add to result with weight
                    actual_h = end_h - start_h
                    actual_w = end_w - start_w
                    
                    # Create weight for this tile (less weight at borders for blending)
                    tile_weight = np.ones((actual_h, actual_w))
                    if overlap > 0:
                        # Reduce weight at borders
                        fade = min(overlap // 2, 32)
                        for k in range(fade):
                            alpha = k / fade
                            if start_h > 0:  # Not first row
                                tile_weight[k, :] *= alpha
                            if end_h < H:    # Not last row
                                tile_weight[-k-1, :] *= alpha
                            if start_w > 0:  # Not first col
                                tile_weight[:, k] *= alpha
                            if end_w < W:    # Not last col
                                tile_weight[:, -k-1] *= alpha
                    
                    # Add weighted result
                    result[start_h:end_h, start_w:end_w] += starless_np * tile_weight[:, :, np.newaxis]
                    weight_map[start_h:end_h, start_w:end_w] += tile_weight
                    
                    # Update progress
                    processed_tiles += 1
                    progress = int((processed_tiles / total_tiles) * 100)
                    self.progress_queue.put(("progress", progress))
        
        # Normalize by weight
        weight_map[weight_map == 0] = 1  # Avoid division by zero
        result = result / weight_map[:, :, np.newaxis]
        
        return result
        
    def get_output_path(self, input_path):
        """Genera path di output con suffisso _starless"""
        path = Path(input_path)
        stem = path.stem
        suffix = path.suffix
        output_name = f"{stem}_starless{suffix}"
        return path.parent / output_name
        
    def check_progress(self):
        """Controlla queue per aggiornamenti progress"""
        try:
            while True:
                item = self.progress_queue.get_nowait()
                msg_type, value, *args = item
                
                if msg_type == "progress":
                    self.progress['value'] = value
                elif msg_type == "status":
                    color = args[0] if args else "black"
                    self.status_label.configure(text=value, foreground=color)
                elif msg_type == "enable_button":
                    self.process_btn.configure(state="normal")
                    
        except queue.Empty:
            pass
            
        # Schedule next check
        self.root.after(100, self.check_progress)


def main():
    root = tk.Tk()
    app = StarlessGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
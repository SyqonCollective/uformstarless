#!/usr/bin/env python3
"""
Setup script per Starless GUI
Installa dipendenze necessarie per M1 Pro
"""

import subprocess
import sys

def install_dependencies():
    """Installa dipendenze per GUI Starless"""
    
    print("🚀 Installing Starless GUI dependencies for M1 Pro...")
    
    dependencies = [
        "torch",           # PyTorch con Metal support
        "torchvision", 
        "Pillow",          # PIL per gestione immagini
        "numpy",
    ]
    
    for dep in dependencies:
        print(f"\n📦 Installing {dep}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    print("\n🎉 All dependencies installed successfully!")
    print("\n🚀 Run the GUI with: python starless_gui.py")
    return True

if __name__ == "__main__":
    install_dependencies()
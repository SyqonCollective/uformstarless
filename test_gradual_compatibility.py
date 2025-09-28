"""
Test rapido della compatibilità Gradual Enhanced UFormer
Verifica che riusciamo a caricare il checkpoint esistente
"""

import torch
import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gradual_compatibility():
    """Test rapido della compatibilità"""
    
    try:
        # Import locale
        from gradual_enhanced_uformer import create_gradually_enhanced_model
        
        logger.info("Testing gradual enhanced model compatibility...")
        
        # Percorsi
        checkpoint_path = "experiments/uformer_debug/checkpoints/best_model.pth"
        
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        # Crea modello enhanced
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        
        model = create_gradually_enhanced_model(
            checkpoint_path=checkpoint_path,
            halo_size=4,
            device=device
        )
        
        logger.info("✓ Model created successfully!")
        logger.info(f"  Model type: {type(model)}")
        logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        logger.info("Testing forward pass...")
        model.eval()
        
        with torch.no_grad():
            # Input di test
            test_input = torch.randn(1, 3, 256, 256).to(device)
            logger.info(f"Test input shape: {test_input.shape}")
            
            # Forward pass
            pred_starless, pred_mask = model(test_input)
            
            logger.info(f"✓ Forward pass successful!")
            logger.info(f"  Starless output: {pred_starless.shape}")
            logger.info(f"  Mask output: {pred_mask.shape}")
            
        logger.info("=" * 50)
        logger.info("✓ GRADUAL ENHANCED MODEL WORKS!")
        logger.info("Ready for gradual fine-tuning")
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == '__main__':
    success = test_gradual_compatibility()
    
    if success:
        print("\n" + "="*50)
        print("READY FOR RUNPOD DEPLOYMENT!")
        print("Files to upload:")
        print("- gradual_enhanced_uformer.py")
        print("- train_gradual_enhanced.py")  
        print("- test_gradual_compatibility.py")
        print("\nRunPod command:")
        print("python test_gradual_compatibility.py")
        print("="*50)
        sys.exit(0)
    else:
        print("\n" + "="*50)
        print("COMPATIBILITY TEST FAILED!")
        print("Need to debug the implementation")
        print("="*50)
        sys.exit(1)


import torch
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
import sys
from pcb11.features import FeatureConfig, extract_features

def test_dino_integration():
    print("Setting up test environment...")
    test_dir = Path("test_dino_output")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    # Create dummy image
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img_path = test_dir / "test_image.jpg"
    img.save(img_path)
    
    print(f"Created dummy image at {img_path}")
    
    # Configure extraction
    config = FeatureConfig(
        image_path=img_path,
        output_dir=test_dir,
        model_name="dinov2_vitb14", # Test vitb14
        device="cpu", # Force CPU for test
        pool_mode="cls",
        batch_size=1
    )
    
    print("Running extract_features...")
    try:
        saved_files = extract_features(config)
        print("Extraction completed!")
        print(f"Saved files: {saved_files.keys()}")
        
        # Verify output shapes
        # Expecting 12 blocks. For 'blocks.11', shape should be [1, 768] (CL token)
        # We requested "cls" pooling in config
        
        passed = True
        for layer, path in saved_files.items():
            data = np.load(path)
            print(f"Layer {layer}: shape={data.shape}")
            
            # shape check: [1, 768] for cls token
            if data.shape != (1, 768):
                print(f"ERROR: Unexpected shape for {layer}. Expected (1, 768), got {data.shape}")
                passed = False
                
        if passed:
            print("\nSUCCESS: DINOv2 integration verified!")
        else:
            print("\nFAILED: Shape mismatch.")
            
    except Exception as e:
        print(f"\nFAILED: Exception during extraction: {e}")
        import traceback
        traceback.print_exc()
    
    # Cleanup
    # shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_dino_integration()

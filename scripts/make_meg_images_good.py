#!/usr/bin/env python3
import re
import shutil
from pathlib import Path

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BAD_MEG_DIR = PROJECT_ROOT / "data/scenes/syns_meg36"
NEW_MEG_DIR = PROJECT_ROOT / "data/scenes/syns_meg36_real"


def resolve_existing_dir(*candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "None of the expected source folders exist: "
        + ", ".join(str(candidate) for candidate in candidates)
    )

def main():
    print("=== Rebuilding MEG Image Set ===")
    good_source_dir = resolve_existing_dir(
        PROJECT_ROOT / "data/scenes/syns_anderson_full",
        PROJECT_ROOT / "data/extracted_anderson_pictures 2",
    )
    
    # 1. Ensure the new destination directory exists (and is completely flat)
    NEW_MEG_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Map all high-quality images by their integer IDs (Scene, View)
    print(f"Scanning {good_source_dir.name} for high-quality source images...")
    good_images = {}
    for p in good_source_dir.glob("*.jpg"):
        match = re.search(r'S(\d+)_Im(\d+)', p.name)
        if match:
            scene_int, view_int = int(match.group(1)), int(match.group(2))
            good_images[(scene_int, view_int)] = p
            
    print(f" -> Found {len(good_images)} valid images in the source folder.\n")

    # 3. Find the target IDs from the bad MEG folder (searching through subfolders)
    print(f"Scanning {BAD_MEG_DIR.name} to identify the 36 test images...")
    meg_image_ids = set()
    for p in BAD_MEG_DIR.rglob("*.jpg"):
        match = re.search(r'S(\d+)_Im(\d+)', p.name)
        if match:
            scene_int, view_int = int(match.group(1)), int(match.group(2))
            meg_image_ids.add((scene_int, view_int))
            
    print(f" -> Identified {len(meg_image_ids)} target images to extract.\n")

    # 4. Copy the good versions into the new flat folder
    print(f"Copying to {NEW_MEG_DIR.name}...")
    success_count = 0
    missing_count = 0
    
    for scene_int, view_int in meg_image_ids:
        if (scene_int, view_int) in good_images:
            src_file = good_images[(scene_int, view_int)]
            # We use the original clean filename from the source folder
            dest_file = NEW_MEG_DIR / src_file.name 
            
            # Copy the file along with its metadata
            shutil.copy2(src_file, dest_file)
            success_count += 1
        else:
            print(f"  [!] ERROR: Could not find S{scene_int}_Im{view_int} in {GOOD_SOURCE_DIR.name}")
            missing_count += 1
            
    print("\n=== Summary ===")
    print(f"Successfully copied: {success_count} images")
    if missing_count > 0:
        print(f"Missing images: {missing_count} (Check your source folder!)")
    else:
        print("All 36 images copied successfully!")

if __name__ == "__main__":
    main()

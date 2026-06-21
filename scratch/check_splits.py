import os
from pathlib import Path

def check_splits():
    splits_dir = Path("D:/codedatgmt/data/splits")
    print(f"Checking splits directory: {splits_dir}\n")
    
    if not splits_dir.exists():
        print("Splits directory does not exist.")
        return
        
    for split in ["train", "val", "test"]:
        img_dir = splits_dir / split / "images"
        if not img_dir.exists():
            print(f"  {split.upper()}: images directory does not exist.")
            continue
            
        all_files = list(img_dir.glob("*.jpg"))
        orig_files = [f for f in all_files if "_orig" in f.name]
        aug_files = [f for f in all_files if "_aug" in f.name]
        other_files = [f for f in all_files if "_orig" not in f.name and "_aug" not in f.name]
        
        print(f"  {split.upper()}:")
        print(f"    Total images: {len(all_files)}")
        print(f"    - Original (_orig): {len(orig_files)}")
        print(f"    - Augmented (_aug): {len(aug_files)}")
        print(f"    - Others: {len(other_files)}")
        
        if len(other_files) > 0:
            print(f"      Example others: {[f.name for f in other_files[:3]]}")
        print()

if __name__ == "__main__":
    check_splits()

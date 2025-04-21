"""
Script to download MS MARCO dataset files from HuggingFace.
This ensures that the data files are available for training without needing to be checked into git.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def download_data():
    """Download MS MARCO dataset from HuggingFace."""
    print("Downloading MS MARCO dataset from HuggingFace...")
    
    # Define paths
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"
    
    # Create directories if they don't exist
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download MS MARCO files
    datasets = ["train", "validation", "test"]
    
    for dataset in datasets:
        output_path = raw_data_dir / f"{dataset}-00000-of-00001.parquet"
        
        # Skip if file already exists
        if output_path.exists():
            print(f"  {dataset} dataset already exists, skipping download")
            continue
        
        try:
            # Download from HuggingFace
            print(f"  Downloading {dataset} dataset...")
            
            # Replace with the actual HuggingFace repo ID and path
            local_file = hf_hub_download(
                repo_id="Apples96/ms_marco_open",
                filename=f"{dataset}-00000-of-00001.parquet",
                local_dir=raw_data_dir
            )
            
            # Move to the correct location if needed
            if Path(local_file) != output_path:
                shutil.move(local_file, output_path)
                
            print(f"  {dataset} dataset downloaded successfully")
            
        except Exception as e:
            print(f"Error downloading {dataset} dataset: {e}")
            return False
    
    # Check for processed files
    required_processed_files = ["queries.json", "passages.json", "matches.json"]
    missing_files = [f for f in required_processed_files if not (processed_data_dir / f).exists()]
    
    if missing_files:
        print(f"\nProcessed files missing: {', '.join(missing_files)}")
        print("Please run the data processing notebook to create these files.")
    else:
        print("\nAll required files are available!")
    
    return True

if __name__ == "__main__":
    success = download_data()
    if success:
        print("\nData download complete!")
    else:
        print("\nData download failed. Please check the error messages above.")
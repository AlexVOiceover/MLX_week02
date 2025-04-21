"""
Script to download MS MARCO dataset files from HuggingFace.
This ensures that the data files are available for training without needing to be checked into git.
"""

import os
import sys
import requests
from pathlib import Path
import shutil
import tempfile

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
            # Download from HuggingFace using direct URL
            print(f"  Downloading {dataset} dataset...")
            
            # Direct URL to the dataset file
            url = f"https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/{dataset}-00000-of-00001.parquet"
            
            # Download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise exception for HTTP errors
            
            # Save the file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                
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
    
    # Clean up any cache folders that might have been created
    cleanup_paths = [
        Path(".cache"),
        Path(tempfile.gettempdir()) / "huggingface"
    ]
    
    for path in cleanup_paths:
        if path.exists() and path.is_dir():
            try:
                print(f"Cleaning up cache directory: {path}")
                shutil.rmtree(path)
            except Exception as e:
                print(f"Warning: Could not clean up {path}: {e}")
    
    return True

if __name__ == "__main__":
    success = download_data()
    if success:
        print("\nData download complete!")
    else:
        print("\nData download failed. Please check the error messages above.")
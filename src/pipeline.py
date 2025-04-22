#!/usr/bin/env python3
"""
Simple pipeline that runs training followed by document indexing.
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Run training and then document indexing."""
    print("Starting pipeline: training + indexing")
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Step 1: Run training
    print("\nRunning training...")
    train_script = project_root / "src" / "training" / "train.py"
    train_result = subprocess.run(["python", str(train_script)], check=False)
    
    if train_result.returncode != 0:
        print("Training failed. Exiting pipeline.")
        sys.exit(train_result.returncode)
    
    # Step 2: Run document indexing (only if training succeeds)
    print("\nRunning document indexing...")
    index_script = project_root / "src" / "indexing" / "document_indexer.py"
    index_result = subprocess.run(["python", str(index_script)], check=False)
    
    if index_result.returncode != 0:
        print("Document indexing failed.")
        sys.exit(index_result.returncode)
    
    print("\nPipeline completed successfully!")
    print("You can now run the search script:")
    print(f"python {project_root}/src/indexing/search.py")

if __name__ == "__main__":
    main()
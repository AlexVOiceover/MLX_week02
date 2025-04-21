# Learn to Search

A Learning to Rank (LTR) system for document retrieval based on triplet loss and pre-trained embeddings.

## Project Overview

This project implements a simple Learning to Rank system that:
1. Uses pre-trained word embeddings from a CBOW model
2. Builds a dual-tower neural ranking model
3. Trains with triplet loss on the MS MARCO dataset
4. Logs experiments using Weights & Biases

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <your-repo-directory>

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Download Data

The project uses the MS MARCO dataset for training. To download the data:

```bash
python scripts/download_data.py
```

This script will download the raw data files from HuggingFace and check if the processed files exist.

## Project Structure

- `src/`
  - `data/`: Data loading and processing code
    - `triplet_dataset.py`: Implements triplet dataset for training
  - `models/`: Model implementations
    - `embeddings.py`: Handles loading pre-trained embeddings
    - `ranking_model.py`: Implements query and document towers
  - `training/`: Training scripts
    - `train.py`: Main training script
  - `utils/`: Utility functions
    - `tokenization.py`: Text tokenization utilities

- `scripts/`: Helper scripts
  - `download_data.py`: Downloads required data files

- `data/`: Data files (gitignored)
  - `raw/`: Raw dataset files
  - `processed/`: Processed JSON files for training

## Training

To train the model:

```bash
python src/training/train.py
```

Training progress will be logged to Weights & Biases.

## License

[MIT License](LICENSE)
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

## Data Preparation

The project uses the MS MARCO dataset for training.

Run the data preparation notebook to download and process the data:

```bash
# If you haven't installed Jupyter yet
pip install jupyter

# Run the notebook
jupyter notebook notebooks/process_msmarco.ipynb
```

This notebook will:
1. Download the MS MARCO dataset from HuggingFace (if not already downloaded)
2. Process the raw data into JSON format:
   - Extract queries, passages, and relevance information
   - Create JSON files required for training (queries.json, passages.json, matches.json)

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

- `notebooks/`
  - `process_msmarco.ipynb`: Downloads and processes MS MARCO data into JSON format

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
# Learn-to-Search

This project implements a neural information retrieval system using the learn-to-rank approach with dual-tower architecture.

## Project Structure

```
MLX_week02/
│
├── data/                   # All data
│   ├── raw/                # Raw data files
│   ├── processed/          # Processed JSON files
│   └── index/              # Vector database
│
├── models/                 # Model storage 
│   ├── saved/              # Local trained models
│   └── wandb/              # Models from Weights & Biases
│
├── notebooks/              # Jupyter notebooks
│   └── process_data.ipynb  # Data processing notebook
│
├── src/                    # Source code (flattened structure)
│   ├── dataset.py          # Dataset handling
│   ├── embeddings.py       # Word embedding functionality
│   ├── model.py            # Neural network models
│   ├── indexer.py          # Document indexing
│   ├── search.py           # Search functionality
│   ├── train.py            # Training logic
│   └── utils.py            # Utility functions
│
├── scripts/                # Runnable scripts
│   ├── main.py             # Main entry point
│   ├── process_data.py     # Data processing script
│   ├── train.py            # Training script
│   ├── index.py            # Indexing script
│   └── search.py           # Search script
│
└── requirements.txt        # Dependencies
```

## Workflow

1. **Data Processing**: Download and prepare MS MARCO dataset
2. **Training**: Train the neural ranking model
3. **Indexing**: Create document vectors and store in ChromaDB
4. **Search**: Retrieve documents using semantic search

## Usage

You can use the main script to run the different parts of the system:

```bash
# Process data
python scripts/main.py process

# Train model
python scripts/main.py train

# Index documents
python scripts/main.py index

# Search
python scripts/main.py search --query "your search query"
```

Or use individual scripts:

```bash
# Process data
python scripts/process_data.py

# Train model
python scripts/train.py

# Index documents
python scripts/index.py

# Search
python scripts/search.py --query "your search query"
```

## Requirements

- Python 3.8+
- PyTorch
- Hugging Face Hub
- ChromaDB
- Weights & Biases

Install dependencies:

```bash
pip install -r requirements.txt
```
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


```bash
# Run Streamlit search app
streamlit run src/search_app.py --server.fileWatcherType none
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

## Configuration

The project uses environment variables from a `.env` file for configuration:

```
# W&B API key
WANDB_API_KEY=your_api_key_here

# Triplet generation configuration
USE_CROSS_QUERY_NEGATIVES=False   # Set to True to use cross-query negative sampling
NEGATIVES_PER_POSITIVE=6          # Number of negative samples per positive (for cross-query strategy)
```

### Negative Sampling Strategies

The system supports two strategies for generating training triplets:

1. **In-Query Negatives (Default)**: Negative examples are taken from the same query's suggested documents.
   - Uses passages that were actually suggested for the query but weren't selected
   - Tends to create "harder" negative examples since they were similar enough to be suggested

2. **Cross-Query Negatives**: Negative examples are randomly sampled from documents associated with other queries.
   - Creates more diverse negative examples
   - May help the model learn broader distinctions between relevant and irrelevant documents
   - Configure using `USE_CROSS_QUERY_NEGATIVES=True` in the `.env` file
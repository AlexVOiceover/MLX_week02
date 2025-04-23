# Learn to Search

A complete end-to-end neural search system using Learning to Rank (LTR) with triplet loss and vector database integration.

## Project Overview

This project implements a modern search system with these components:

1. **Neural Ranking Model**: A dual-tower architecture trained on the MS MARCO dataset using triplet loss
2. **Vector Database**: ChromaDB for efficient similarity search 
3. **Experiment Tracking**: Weights & Biases integration for model tracking
4. **Search Interface**: Interactive command-line search functionality

The system demonstrates the entire pipeline from data processing to search inference.

## System Architecture

The system consists of three main components:

### 1. Ranking Model (Training Phase)
- **Dual-Tower Architecture**: Separate encoders for queries and documents
- **Pre-trained Embeddings**: Uses frozen word embeddings as a foundation
- **Triplet Loss**: Trains by contrasting query-positive and query-negative pairs
- **Experiment Tracking**: Logs to Weights & Biases for visualization

### 2. Document Indexing
- **Model Loading**: Retrieves trained model from Weights & Biases or local storage
- **Batch Processing**: Efficiently encodes documents with trained document tower
- **Vector Storage**: Saves document embeddings in ChromaDB with metadata

### 3. Search Functionality
- **Query Encoding**: Transforms search queries into embeddings using query tower
- **Vector Similarity**: Finds closest document vectors in ChromaDB using cosine similarity
- **Result Ranking**: Returns most similar documents with relevance scores

## Workflow

1. **Data Preparation**:
   - Download MS MARCO dataset
   - Process into JSON format for training

2. **Model Training**:
   - Load pre-trained word embeddings
   - Train ranking model with triplet loss
   - Save model artifacts to Weights & Biases

3. **Document Indexing**:
   - Load trained model
   - Encode documents and store in ChromaDB
   - Prepare for efficient search

4. **Search**:
   - Accept user queries
   - Encode with query tower
   - Retrieve relevant documents from ChromaDB
   - Display results with similarity scores

## Repository Structure

```
├── data/                      # Data directory (gitignored)
│   ├── raw/                   # Raw dataset files
│   ├── processed/             # Processed JSON files
│   └── chroma/                # ChromaDB vector database
│
├── notebooks/
│   └── process_msmarco.ipynb  # Data downloading and processing
│
├── src/
│   ├── data/                  # Data handling
│   │   └── triplet_dataset.py # Dataset implementation
│   │
│   ├── indexing/              # Indexing and search
│   │   ├── document_indexer.py # Document encoding and storage
│   │   └── search.py          # Interactive search interface
│   │
│   ├── models/                # Model definitions
│   │   ├── embeddings.py      # Pre-trained embedding loading
│   │   └── ranking_model.py   # Query and document towers
│   │
│   ├── training/              # Training code
│   │   └── train.py           # Main training script
│   │
│   └── utils/                 # Utilities
│       └── tokenization.py    # Text processing functions
│
├── models/                    # Model storage (gitignored)
│   ├── saved/                 # Locally saved models
│   └── wandb/                 # Models from Weights & Biases
│
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Key File Descriptions

### Data Processing
- **process_msmarco.ipynb**: Downloads and processes the MS MARCO dataset into JSON format for training. Creates queries.json, passages.json, and matches.json.

### Model Architecture
- **embeddings.py**: Loads pre-trained CBOW embeddings from Hugging Face and prepares them for use in the ranking model.
- **ranking_model.py**: Implements the dual-tower architecture with QueryTower and DocumentTower classes, plus the triplet loss function.

### Training
- **triplet_dataset.py**: Handles data loading and triplet generation for training.
- **train.py**: Orchestrates the training process, including model setup, optimization, logging to W&B, and model saving.

### Indexing and Search
- **document_indexer.py**: Loads the trained document tower, encodes passages, and stores them in ChromaDB.
- **search.py**: Provides interactive command-line search functionality using the trained query tower.

### Utilities
- **tokenization.py**: Contains text processing utilities for tokenization, token-to-ID conversion, and padding.

## Setup and Usage

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd learn-to-search

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Run the data processing notebook to download and prepare the MS MARCO dataset:

```bash
jupyter notebook notebooks/process_msmarco.ipynb
```

This creates three JSON files in the data/processed directory:
- queries.json: Search queries
- passages.json: Document passages
- matches.json: Relevance information

### Training

Train the ranking model:

```bash
python src/training/train.py
```

This process:
1. Loads pre-trained embeddings
2. Creates the dual-tower model
3. Trains using triplet loss
4. Logs metrics to Weights & Biases
5. Saves the trained model

### Indexing

Index documents into ChromaDB:

```bash
python src/indexing/document_indexer.py
```

This process:
1. Loads the trained model (from W&B or local)
2. Encodes all passages
3. Stores embeddings in ChromaDB

### Searching

Run the interactive search interface:

```bash
# Interactive mode
python src/indexing/search.py

# Pass a query directly
python src/indexing/search.py --query "your search query"
```

Type a search query to see relevant passages ranked by similarity. The search results include:

- Document ID
- Similarity score (using cosine similarity)
- Preview of the document text

## W&B Integration

The project integrates with Weights & Biases for:
- Training metrics visualization
- Model artifact storage and versioning
- Experiment tracking

To use W&B features:
1. Create a .env file with your W&B API key: `WANDB_API_KEY=your_key_here`
2. All scripts will automatically use this key for authentication

## Extending the Project

Some ways to extend this project:
- Add GPU acceleration for faster training and inference
- Implement more sophisticated neural architectures
- Create a web UI for the search interface
- Add evaluation metrics for search quality
- Expand to other datasets beyond MS MARCO

## License

[MIT License](LICENSE)
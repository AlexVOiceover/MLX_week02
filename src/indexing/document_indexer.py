import torch
import json
import sys
from pathlib import Path
import chromadb
import wandb
import os
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our modules
from src.models.embeddings import load_pretrained_embedding
from src.models.ranking_model import DocumentTower
from src.utils.tokenization import tokenize_text, convert_to_token_ids, pad_or_truncate


def load_model():
    """Load the trained document tower model from Weights & Biases"""
    print("Loading model from Weights & Biases...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize W&B
    api = wandb.Api()
    
    # Get the most recent model artifact
    project_name = "learn-to-search"  # The project name used in train.py
    
    try:
        # Query artifacts of type "model"
        artifacts = api.artifacts(project_name + "/model", type="model")
        
        if not artifacts:
            print("No model artifacts found in W&B. Falling back to local model.")
            # Fallback to local model
            model_dir = Path(__file__).parent.parent.parent / "models" / "saved"
            model_path = list(model_dir.glob("ranking_model_*.pt"))[0]
            print(f"Loading local model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
        else:
            # Get latest model artifact
            latest_artifact = artifacts[0]
            print(f"Found model artifact: {latest_artifact.name}")
            
            # Download the artifact
            model_dir = Path(__file__).parent.parent.parent / "models" / "wandb"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Download the artifact files
            latest_artifact.download(root=str(model_dir))
            
            # Find the model file in the downloaded artifact
            model_files = list(model_dir.glob("**/*.pt"))
            model_path = model_files[0]
            print(f"Downloaded model to {model_path}")
            
            # Load the checkpoint
            checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Error accessing W&B: {e}")
        print("Falling back to local model...")
        model_dir = Path(__file__).parent.parent.parent / "models" / "saved"
        model_path = list(model_dir.glob("ranking_model_*.pt"))[0]
        checkpoint = torch.load(model_path, map_location=device)
    
    # Load embedding layer and document tower
    embedding_layer, word2idx, _ = load_pretrained_embedding()
    doc_tower = DocumentTower(embedding_layer)
    
    # Load the document tower weights
    doc_tower.load_state_dict(checkpoint['doc_tower'])
    
    # Move model to device
    doc_tower = doc_tower.to(device)
    
    # Set to evaluation mode
    doc_tower.eval()
    
    return doc_tower, word2idx, device


def process_batch(doc_texts, doc_ids, doc_tower, word2idx, device, max_length=64):
    """Process a batch of documents at once"""
    batch_size = len(doc_texts)
    
    # Initialize tensor to hold all token IDs
    all_tokens = []
    
    # Tokenize all documents in the batch
    for doc_text in doc_texts:
        tokens = tokenize_text(doc_text)
        token_ids = convert_to_token_ids(tokens, word2idx)
        token_ids = pad_or_truncate(token_ids, max_length)
        all_tokens.append(token_ids)
    
    # Convert to tensor and move to device
    batch_tensor = torch.tensor(all_tokens).to(device)
    
    # Encode with document tower
    with torch.no_grad():
        batch_vectors = doc_tower(batch_tensor)
    
    # Convert to list format for ChromaDB
    vector_list = batch_vectors.cpu().tolist()
    
    # Prepare results for ChromaDB
    return {
        "ids": doc_ids,
        "embeddings": vector_list,
        "metadatas": [{"text": text} for text in doc_texts]
    }


def index_documents():
    """Main function to encode documents and store in ChromaDB"""
    # 1. Load the model
    doc_tower, word2idx, device = load_model()
    
    # 2. Load documents
    data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
    with open(data_dir / "passages.json", 'r') as f:
        passages = json.load(f)
    print(f"Loaded {len(passages)} documents")
    
    # 3. Initialize ChromaDB
    chroma_dir = Path(__file__).parent.parent.parent / "data" / "chroma"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    # Create or get collection
    try:
        client.delete_collection("passages")
        print("Deleted existing collection.")
    except Exception as e:
        print(f"Collection didn't exist yet: {e}")
    
    collection = client.create_collection("passages")
    print("Created ChromaDB collection: passages")
    
    # 4. Process and index documents in batches
    batch_size = 64  # Process 64 documents at a time
    total_docs = len(passages)
    
    # Convert dictionary to lists for batch processing
    all_ids = list(passages.keys())
    all_texts = list(passages.values())
    
    # Process in batches with progress bar
    indexed_count = 0
    for batch_start in tqdm(range(0, total_docs, batch_size), desc="Indexing batches"):
        batch_end = min(batch_start + batch_size, total_docs)
        
        # Get batch of documents
        batch_ids = all_ids[batch_start:batch_end]
        batch_texts = all_texts[batch_start:batch_end]
        
        # Process batch
        batch_data = process_batch(batch_texts, batch_ids, doc_tower, word2idx, device)
        
        # Add to ChromaDB
        collection.add(
            ids=batch_data["ids"],
            embeddings=batch_data["embeddings"],
            metadatas=batch_data["metadatas"]
        )
        
        indexed_count += len(batch_ids)
    
    print(f"Successfully indexed {indexed_count} documents in ChromaDB")
    
    # 5. Test search (optional)
    print("Running test search...")
    test_vector_size = batch_data["embeddings"][0]
    test_results = collection.query(
        query_embeddings=[[0.1] * len(test_vector_size)],  # Random query vector for testing
        n_results=2
    )
    print(f"Test search returned {len(test_results['ids'][0])} results")


if __name__ == "__main__":
    index_documents()
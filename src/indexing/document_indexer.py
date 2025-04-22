import torch
import json
import sys
from pathlib import Path
import chromadb
import wandb
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our modules
from src.models.embeddings import load_pretrained_embedding
from src.models.ranking_model import DocumentTower
from src.utils.tokenization import tokenize_text, convert_to_token_ids, pad_or_truncate


def load_model():
    """Load the trained document tower model from Weights & Biases"""
    print("Loading model from Weights & Biases...")
    
    # Initialize W&B
    api = wandb.Api()
    
    # Get the most recent model artifact
    project_name = "learn-to-search"  # The project name used in train.py
    
    # Query artifacts of type "model"
    artifacts = api.artifacts(project_name + "/model", type="model")
    
    if not artifacts:
        print("No model artifacts found in W&B. Falling back to local model.")
        # Fallback to local model
        model_dir = Path(__file__).parent.parent.parent / "models" / "saved"
        model_path = list(model_dir.glob("ranking_model_*.pt"))[0]
        print(f"Loading local model from {model_path}")
        checkpoint = torch.load(model_path)
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
        checkpoint = torch.load(model_path)
    
    # Load embedding layer and document tower
    embedding_layer, word2idx, _ = load_pretrained_embedding()
    doc_tower = DocumentTower(embedding_layer)
    
    # Load the document tower weights
    doc_tower.load_state_dict(checkpoint['doc_tower'])
    
    # Set to evaluation mode
    doc_tower.eval()
    
    return doc_tower, word2idx


def encode_document(doc_text, doc_tower, word2idx, max_length=64):
    """Convert a document text into a vector using the document tower"""
    # Tokenize and convert to tensor
    tokens = tokenize_text(doc_text)
    token_ids = convert_to_token_ids(tokens, word2idx)
    token_ids = pad_or_truncate(token_ids, max_length)
    doc_tensor = torch.tensor(token_ids).unsqueeze(0)  # Add batch dimension
    
    # Encode with document tower
    with torch.no_grad():
        doc_vector = doc_tower(doc_tensor)
    
    return doc_vector.squeeze().tolist()  # Convert to list for storage


def index_documents():
    """Main function to encode documents and store in ChromaDB"""
    # 1. Load the model
    doc_tower, word2idx = load_model()
    
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
    except:
        pass  # Collection didn't exist yet
    
    collection = client.create_collection("passages")
    print("Created ChromaDB collection: passages")
    
    # 4. Process and index all documents
    doc_count = 0
    
    for doc_id, doc_text in passages.items():
        # Encode document
        doc_vector = encode_document(doc_text, doc_tower, word2idx)
        
        # Add to ChromaDB
        collection.add(
            ids=[doc_id],
            embeddings=[doc_vector],
            metadatas=[{"text": doc_text}]
        )
        
        doc_count += 1
        if doc_count % 100 == 0:
            print(f"Indexed {doc_count} documents")
    
    print(f"Successfully indexed {doc_count} documents in ChromaDB")
    
    # 5. Test search (optional)
    test_results = collection.query(
        query_embeddings=[[0.1] * len(doc_vector)],  # Random query vector for testing
        n_results=2
    )
    print(f"Test search returned {len(test_results['ids'][0])} results")


if __name__ == "__main__":
    index_documents()
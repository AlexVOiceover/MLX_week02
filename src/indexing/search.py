import torch
import sys
from pathlib import Path
import chromadb
import wandb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our modules
from src.models.embeddings import load_pretrained_embedding
from src.models.ranking_model import QueryTower
from src.utils.tokenization import tokenize_text, convert_to_token_ids, pad_or_truncate


def load_query_model():
    """Load the trained query tower model from Weights & Biases"""
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
    
    # Load embedding layer and query tower
    embedding_layer, word2idx, _ = load_pretrained_embedding()
    query_tower = QueryTower(embedding_layer)
    
    # Load the query tower weights
    query_tower.load_state_dict(checkpoint['query_tower'])
    
    # Set to evaluation mode
    query_tower.eval()
    
    return query_tower, word2idx


def search():
    """Simple search function to search the indexed documents"""
    # 1. Load the query tower from W&B
    query_tower, word2idx = load_query_model()
    
    # 2. Connect to ChromaDB
    print("Connecting to ChromaDB...")
    chroma_dir = Path(__file__).parent.parent.parent / "data" / "chroma"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_collection("passages")
    
    # 3. Get user query
    query_text = input("Enter your search query: ")
    
    # 4. Process and encode query
    print("Encoding query...")
    # Tokenize
    tokens = tokenize_text(query_text)
    # Convert to token IDs
    token_ids = convert_to_token_ids(tokens, word2idx)
    # Pad or truncate to fixed length
    max_query_length = 20
    token_ids = pad_or_truncate(token_ids, max_query_length)
    # Convert to tensor
    query_tensor = torch.tensor(token_ids).unsqueeze(0)  # Add batch dimension
    
    # Encode with query tower
    with torch.no_grad():
        query_vector = query_tower(query_tensor)
    
    # 5. Search in ChromaDB
    print("Searching documents...")
    results = collection.query(
        query_embeddings=[query_vector.squeeze().tolist()],
        n_results=5
    )
    
    # 6. Display results
    print(f"\nSearch results for: '{query_text}'")
    print("-" * 50)
    
    for i in range(len(results['ids'][0])):
        doc_id = results['ids'][0][i]
        text = results['metadatas'][0][i]['text']
        distance = results['distances'][0][i]
        
        print(f"Result {i+1}: Document ID {doc_id}")
        print(f"Similarity: {distance}")
        # Show the beginning of the text (truncated if too long)
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"Text: {preview}")
        print("-" * 50)


if __name__ == "__main__":
    search()
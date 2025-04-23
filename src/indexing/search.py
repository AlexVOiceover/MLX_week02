import torch
import sys
import os
from pathlib import Path
import chromadb
import wandb
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
load_dotenv()

# Import our modules
from src.models.embeddings import load_pretrained_embedding
from src.models.ranking_model import QueryTower
from src.utils.tokenization import tokenize_text, convert_to_token_ids, pad_or_truncate


def load_query_model():
    """Load the trained query tower model from Weights & Biases"""
    print("Loading model from Weights & Biases...")
    
    # Set device for consistent behavior
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dynamic W&B artifact retrieval
    try:
        # Initialize W&B API
        api = wandb.Api()
        
        # Define project and entity
        entity = os.getenv("WANDB_ENTITY", "alexvoiceover-fac")  # Get from env or use default
        project_name = "learn-to-search"
        
        print(f"Looking for model artifacts in {entity}/{project_name}")
        
        # Get runs that have model artifacts
        runs = api.runs(f"{entity}/{project_name}")
        
        # Find all model artifacts
        latest_model_artifact = None
        
        # Start a new run for artifact usage
        with wandb.init(project=project_name, job_type="search", name="search-run") as run:
            # Find the latest model artifact from any run
            for wandb_run in runs:
                # Get artifacts of type 'model' from this run
                for artifact in wandb_run.logged_artifacts():
                    if artifact.type == 'model':
                        print(f"Found model artifact: {artifact.name}")
                        latest_model_artifact = artifact.name
                        # We only need the latest one
                        break
                if latest_model_artifact:
                    break
            
            if latest_model_artifact:
                # Use the found artifact
                print(f"Using model artifact: {latest_model_artifact}")
                artifact = run.use_artifact(f"{entity}/{project_name}/{latest_model_artifact}", type='model')
                
                # Download the artifact
                model_dir = Path(__file__).parent.parent.parent / "models" / "wandb"
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Download the artifact files
                artifact_dir = artifact.download(root=str(model_dir))
                print(f"Downloaded artifact to {artifact_dir}")
                
                # Find the model file in the downloaded artifact
                model_files = list(Path(artifact_dir).glob("*.pt"))
                if not model_files:
                    model_files = list(Path(artifact_dir).glob("**/*.pt"))  # Try recursive search
                    
                if model_files:
                    model_path = model_files[0]
                    print(f"Found model file: {model_path}")
                    
                    # Load the checkpoint
                    checkpoint = torch.load(model_path, map_location=device)
                else:
                    raise FileNotFoundError("No model file found in artifact")
            else:
                raise ValueError("No model artifacts found in any run")
                
    except Exception as e:
        print(f"Error with W&B artifact: {e}")
        print("Falling back to local model...")
        # Fallback to local model
        model_dir = Path(__file__).parent.parent.parent / "models" / "saved"
        model_path = list(model_dir.glob("ranking_model_*.pt"))[0]
        print(f"Loading local model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
    
    # Load embedding layer and query tower
    embedding_layer, word2idx, _ = load_pretrained_embedding()
    query_tower = QueryTower(embedding_layer)
    
    # Load the query tower weights
    query_tower.load_state_dict(checkpoint['query_tower'])
    
    # Move model to device
    query_tower = query_tower.to(device)
    
    # Set to evaluation mode
    query_tower.eval()
    
    return query_tower, word2idx, device


def search():
    """Simple search function to search the indexed documents"""
    # 1. Load the query tower from W&B
    query_tower, word2idx, device = load_query_model()
    
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
    # Convert to tensor and move to device
    query_tensor = torch.tensor(token_ids).unsqueeze(0).to(device)  # Add batch dimension
    
    # Encode with query tower
    with torch.no_grad():
        query_vector = query_tower(query_tensor)
    
    # Get vector on CPU for ChromaDB
    query_vector_cpu = query_vector.cpu()
    
    # 5. Search in ChromaDB
    print("Searching documents...")
    results = collection.query(
        query_embeddings=[query_vector_cpu.squeeze().tolist()], 
        n_results=5
    )
    
    # 6. Display results
    print(f"\nSearch results for: '{query_text}'")
    print("-" * 50)
    
    if not results["ids"][0]:
        print("No results found. Try a different query.")
        return
    
    for i in range(len(results["ids"][0])):
        doc_id = results["ids"][0][i]
        text = results["metadatas"][0][i]["text"]
        distance = results["distances"][0][i]
        
        print(f"Result {i+1}: Document ID {doc_id}")
        print(f"Similarity: {distance}")
        # Show the beginning of the text (truncated if too long)
        preview = text[:200] + "..." if len(text) > 200 else text
        print(f"Text: {preview}")
        print("-" * 50)
    
    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    search()
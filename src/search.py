import torch
import sys
import os
import argparse
from pathlib import Path
import chromadb
import wandb
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
load_dotenv()

# Import our modules
from embeddings import load_pretrained_embedding
from model import QueryTower
from utils import tokenize_text, convert_to_token_ids, pad_or_truncate


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
        model_artifacts = []
        
        # Start a new run for artifact usage
        with wandb.init(project=project_name, job_type="search", name="search-run") as run:
            # Find latest model artifact
            model_artifacts = []
            
            # Get all artifacts of type 'model' from all runs
            for wandb_run in runs:
                for artifact in wandb_run.logged_artifacts():
                    if artifact.type == 'model':
                        name = artifact.name
                        # Extract timestamp from ranking_model_YYYYMMDD-HHMMSS
                        if "ranking_model_" in name:
                            model_artifacts.append(name)
            
            # Sort by timestamp (part after ranking_model_)
            model_artifacts.sort(key=lambda x: x.split("ranking_model_")[1].split(":")[0])
            
            # Get the last one (latest timestamp)
            latest_model_artifact = model_artifacts[-1] if model_artifacts else None
            
            if latest_model_artifact:
                # Extract timestamp for display
                timestamp = latest_model_artifact.split("ranking_model_")[1].split(":")[0]
                print(f"Latest model: {latest_model_artifact} (timestamp: {timestamp})")
            
            if latest_model_artifact:
                # Use the found artifact
                print(f"Using model artifact: {latest_model_artifact}")
                artifact = run.use_artifact(f"{entity}/{project_name}/{latest_model_artifact}", type='model')
                
                # First, clear any existing downloads in the wandb directory to prevent caching issues
                model_dir = Path(__file__).parent.parent / "models" / "wandb"
                
                # Remove previously downloaded artifacts to avoid caching issues
                if model_dir.exists():
                    import shutil
                    print(f"Cleaning wandb model directory: {model_dir}")
                    shutil.rmtree(model_dir)
                
                # Create a fresh directory
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Download the artifact files
                print(f"Downloading fresh copy of artifact to {model_dir}...")
                artifact_dir = artifact.download(root=str(model_dir))
                print(f"Downloaded artifact to {artifact_dir}")
                
                # Find the model file in the downloaded artifact
                model_files = list(Path(artifact_dir).glob("*.pt"))
                if not model_files:
                    model_files = list(Path(artifact_dir).glob("**/*.pt"))  # Try recursive search
                    
                if model_files:
                    model_path = model_files[0]
                    print(f"Found model file: {model_path}")
                    
                    # Verify we got the right model version by checking the filename
                    model_filename = model_path.name
                    expected_timestamp = latest_model_artifact.split("ranking_model_")[1].split(":")[0]
                    
                    if expected_timestamp not in str(model_path):
                        print(f"WARNING: Downloaded model filename {model_filename} doesn't match expected artifact timestamp {expected_timestamp}")
                        print(f"This could indicate a caching issue with W&B.")
                    else:
                        print(f"Verified model file matches expected artifact timestamp: {expected_timestamp}")
                    
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
        model_dir = Path(__file__).parent.parent / "models" / "saved"
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


def search(query_text=None):
    """Simple search function to search the indexed documents"""
    # 1. Load the query tower from W&B
    query_tower, word2idx, device = load_query_model()
    
    # 2. Connect to ChromaDB
    print("Connecting to ChromaDB...")
    chroma_dir = Path(__file__).parent.parent / "data" / "index"
    client = chromadb.PersistentClient(path=str(chroma_dir))
    
    # Try to get the collection, recreate with cosine similarity if needed
    try:
        collection = client.get_collection("passages")
        # Check if collection has cosine similarity configured
        if collection.metadata.get("hnsw:space") != "cosine":
            print("Warning: Collection does not use cosine similarity. Search results may be inaccurate.")
            print("Consider reindexing using document_indexer.py")
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return
    
    # 3. Get user query
    if query_text is None:
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
    
    # Check if results make sense - if they don't, the similarity metric might be wrong
    if results["distances"] and results["distances"][0]:
        first_distance = results["distances"][0][0]
        if first_distance > 1.5:  # Cosine similarity should be between -1 and 1
            print(f"Warning: Distance values ({first_distance}) suggest incorrect similarity metric.")
            print("Consider reindexing documents with document_indexer.py")
    
    # 6. Display results
    print(f"\nSearch results for: '{query_text}'")
    print("-" * 50)
    
    if not results["ids"][0]:
        print("No results found. Try a different query.")
        return
    
    # Get the sorted indices (sort by distance: smaller = more similar)
    sorted_indices = sorted(range(len(results["distances"][0])), key=lambda i: results["distances"][0][i])

    # Display results in sorted order
    for rank, i in enumerate(sorted_indices):
        doc_id = results["ids"][0][i]
        text = results["metadatas"][0][i]["text"]
        distance = results["distances"][0][i]
        
        print(f"Result {rank+1}: Document ID {doc_id}")
        # For cosine similarity, the distance value from ChromaDB is (1 - cosine_similarity)
        # So we need to convert it back to a similarity score
        similarity_score = 1.0 - distance if distance <= 2.0 else "N/A (incorrect metric)"
        
        # Display similarity score
        if isinstance(similarity_score, float):
            print(f"Similarity: {similarity_score:.4f} (ChromaDB distance: {distance:.4f})")
        else:
            print(f"Similarity: {similarity_score} (ChromaDB distance: {distance:.4f})")
        # Show the full text
        print(f"Text: {text}")
        print("-" * 50)
    
    # Finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Search indexed documents.')
    parser.add_argument('--query', '-q', type=str, 
                        help='Search query text. If not provided, will prompt for input.')
    args = parser.parse_args()
    
    # Call search with the provided query, if any
    search(args.query)
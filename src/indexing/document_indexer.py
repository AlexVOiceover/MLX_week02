import torch
import json
import sys
from pathlib import Path
import chromadb
import wandb
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file
load_dotenv()

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
        with wandb.init(project=project_name, job_type="inference", name="indexer-run") as run:
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
    
    # Set up a single progress bar for all documents
    print(f"Indexing {total_docs} documents in batches of {batch_size}...")
    
    # Create a progress bar for documents
    progress_bar = tqdm(total=total_docs, desc="Documents indexed", unit="docs")
    
    # Process in batches
    indexed_count = 0
    start_time = time.time()
    
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        current_batch_size = batch_end - batch_start
        
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
        
        # Update counters and progress
        indexed_count += current_batch_size
        progress_bar.update(current_batch_size)
        
        # Occasionally log progress (every 10% or so)
        if indexed_count % max(total_docs // 10, batch_size) < batch_size:
            elapsed = time.time() - start_time
            docs_per_sec = indexed_count / elapsed
            estimated_total = elapsed / indexed_count * total_docs
            remaining = estimated_total - elapsed
            
            # Update progress bar description with rate info
            progress_bar.set_description(
                f"Indexing: {docs_per_sec:.1f} docs/sec"
            )
    
    # Close progress bar
    progress_bar.close()
    
    # Log final stats
    total_time = time.time() - start_time
    print(f"Successfully indexed {indexed_count} documents in {total_time:.1f} seconds")
    print(f"Average indexing rate: {indexed_count/total_time:.1f} documents/second")
    
    # 5. Test search (optional)
    print("Running test search...")
    test_vector_size = len(batch_data["embeddings"][0])
    test_results = collection.query(
        query_embeddings=[[0.1] * test_vector_size],  # Random query vector for testing
        n_results=2
    )
    print(f"Test search returned {len(test_results['ids'][0])} results")


if __name__ == "__main__":
    index_documents()
import torch
import sys
from pathlib import Path
import chromadb

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import our modules
from src.models.embeddings import load_pretrained_embedding
from src.models.ranking_model import QueryTower
from src.utils.tokenization import tokenize_text, convert_to_token_ids, pad_or_truncate


def search():
    """Simple search function to search the indexed documents"""
    # 1. Load embedding and query tower
    print("Loading model...")
    embedding_layer, word2idx, _ = load_pretrained_embedding()
    query_tower = QueryTower(embedding_layer)
    
    # Find the latest model file
    model_dir = Path(__file__).parent.parent.parent / "models" / "saved"
    model_path = list(model_dir.glob("ranking_model_*.pt"))[0]
    
    # Load trained model weights
    checkpoint = torch.load(model_path)
    query_tower.load_state_dict(checkpoint['query_tower'])
    query_tower.eval()  # Set to evaluation mode
    
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
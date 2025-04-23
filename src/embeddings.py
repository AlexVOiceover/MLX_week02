import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download


def load_pretrained_embedding():
    """
    Load a pre-trained embedding model from Hugging Face.

    Returns:
        embedding: PyTorch embedding layer
        word2idx: Dictionary mapping words to indices
        idx2word: Dictionary mapping indices to words
    """
    # Download model from HuggingFace
    model_file = hf_hub_download(
        repo_id="Apples96/cbow_model", filename="output/cbow_model_full.pt"
    )

    # Load the checkpoint
    checkpoint = torch.load(model_file)

    # Extract vocabulary mappings
    word2idx = checkpoint["token_to_index"]
    idx2word = checkpoint["index_to_token"]

    # Get model dimensions
    vocab_size = len(word2idx)
    embed_dim = checkpoint["embedding_dim"]

    # Create an embedding layer with the pre-trained weights
    embedding_layer = nn.Embedding(vocab_size, embed_dim)

    # Load weights from checkpoint - find the embedding weights
    model_dict = checkpoint["model_state_dict"]
    embedding_weights = model_dict["embedding.weight"]
    embedding_layer.weight.data.copy_(embedding_weights)

    # Freeze the embedding layer
    embedding_layer.weight.requires_grad = False

    print(f"Loaded embedding model with {vocab_size} words, dimension {embed_dim}")

    return embedding_layer, word2idx, idx2word


if __name__ == "__main__":
    # Simple test
    embedding, word2idx, idx2word = load_pretrained_embedding()

    # Check a few sample words
    for word in ["the", "and", "learning"]:
        if word in word2idx:
            idx = word2idx[word]
            print(f"'{word}' has index {idx}")

            # Convert to embedding
            word_tensor = torch.tensor([idx])
            vector = embedding(word_tensor)
            print(f"Embedding shape: {vector.shape}")
        else:
            print(f"'{word}' not in vocabulary")

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryTower(nn.Module):
    """Simple encoder for queries."""
    def __init__(self, embedding_layer):
        super(QueryTower, self).__init__()
        self.embedding = embedding_layer
        
        # Get embedding dimension from the embedding layer
        embedding_dim = self.embedding.embedding_dim
        
        # Add a simple trainable linear layer
        self.transform = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        # Get embeddings for each token
        embedded = self.embedding(x)
        
        # Average pooling (mean of all token embeddings)
        pooled = torch.mean(embedded, dim=1)
        
        # Apply trainable transformation
        transformed = self.transform(pooled)
        
        return transformed


class DocumentTower(nn.Module):
    """Simple encoder for documents (same architecture as query tower)."""
    def __init__(self, embedding_layer):
        super(DocumentTower, self).__init__()
        self.embedding = embedding_layer
        
        # Get embedding dimension from the embedding layer
        embedding_dim = self.embedding.embedding_dim
        
        # Add a simple trainable linear layer
        self.transform = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        # Get embeddings for each token
        embedded = self.embedding(x)
        
        # Average pooling (mean of all token embeddings)
        pooled = torch.mean(embedded, dim=1)
        
        # Apply trainable transformation
        transformed = self.transform(pooled)
        
        return transformed


def triplet_loss(query_vec, pos_vec, neg_vec, margin=0.2):
    """
    Simple triplet loss implementation.
    
    Args:
        query_vec: Query embedding vector
        pos_vec: Positive document embedding vector
        neg_vec: Negative document embedding vector
        margin: Margin to enforce between positive and negative distances
        
    Returns:
        loss: The triplet loss value
    """
    # Calculate cosine similarities
    sim_pos = F.cosine_similarity(query_vec, pos_vec)
    sim_neg = F.cosine_similarity(query_vec, neg_vec)
    
    # Calculate difference (we want pos_sim > neg_sim)
    diff = sim_pos - sim_neg
    
    # Apply margin - we want diff to be at least 'margin'
    # This is the max(0, margin - diff) function
    zeros = torch.zeros_like(diff)  # Create tensor of zeros with same shape
    losses = margin - diff  # Calculate margin - diff for each pair
    
    # Take max(0, margin - diff) for each element
    # If margin - diff < 0, use 0 (zero loss)
    # If margin - diff > 0, use margin - diff (there's a violation of the margin)
    loss = torch.max(zeros, losses)
    
    # If working with batches, take mean loss across batch
    if loss.dim() > 0:
        loss = loss.mean()
    
    return loss


# Simple test
if __name__ == "__main__":
    # Create dummy embedding layer
    vocab_size = 100
    embedding_dim = 50
    embedding = nn.Embedding(vocab_size, embedding_dim)
    
    # Create towers
    query_tower = QueryTower(embedding)
    doc_tower = DocumentTower(embedding)
    
    # Sample data (batch_size=2, sequence_length=5)
    query = torch.randint(0, vocab_size, (2, 5))  # 2 queries, 5 tokens each
    pos_doc = torch.randint(0, vocab_size, (2, 5))  # 2 positive docs
    neg_doc = torch.randint(0, vocab_size, (2, 5))  # 2 negative docs
    
    # Forward pass
    query_vec = query_tower(query)
    pos_vec = doc_tower(pos_doc)
    neg_vec = doc_tower(neg_doc)
    
    # Calculate loss
    loss = triplet_loss(query_vec, pos_vec, neg_vec)
    
    print(f"Query shape: {query.shape}")
    print(f"Query vector shape: {query_vec.shape}")
    print(f"Loss: {loss.item()}")
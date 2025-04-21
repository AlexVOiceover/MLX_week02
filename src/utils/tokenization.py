import numpy as np
from collections import Counter
import os
import re

# Remove the import from datasettext_build as we don't need it

def tokenize_text(text):
    """
    Tokenize text by:
    1. Converting to lowercase
    2. Replacing non-alphanumeric characters with spaces
    3. Splitting on whitespace
    
    Returns a list of tokens (words).
    """
    # Convert to lowercase
    text = text.lower()
    
    # Replace any character that is not a letter or number with a space
    text = re.sub(r'[^a-z0-9]', ' ', text)
    
    # Split on whitespace and filter empty tokens
    tokens = [token for token in text.split() if token]
    
    return tokens

def build_vocabulary(tokens, min_count=5):
    """
    Build a vocabulary from tokens.
    Returns:
        - word2idx: A dictionary mapping words to indices
        - idx2word: A dictionary mapping indices to words
        - word_counts: A Counter object with word frequencies
    """
    # Count word frequencies
    word_counts = Counter(tokens)
    
    # Filter words that appear less than min_count times
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    
    # Create word-to-index mapping
    # We'll add special tokens:
    # - <PAD> for padding sequences
    # - <UNK> for unknown words (words not in our vocabulary)
    word2idx = {'<PAD>': 0, '<UNK>': 1}
    for i, word in enumerate(vocab):
        word2idx[word] = i #+ 2  # +2 because we have 2 special tokens
    
    # Create index-to-word mapping
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    print(f"Vocabulary size: {len(word2idx)} words")
    
    return word2idx, idx2word, word_counts

def convert_to_token_ids(tokens, word2idx):
    """
    Convert tokens to token IDs using the vocabulary.
    If a token is not in the vocabulary, use the <UNK> token ID.
    """
    token_ids = [word2idx.get(token, word2idx.get('<UNK>', 0)) for token in tokens]
    
    return token_ids

def pad_or_truncate(token_ids, max_length, pad_token_id=0):
    """
    Pad or truncate a list of token IDs to the specified length.
    
    Args:
        token_ids: List of token IDs
        max_length: Target length
        pad_token_id: ID to use for padding (default: 0)
        
    Returns:
        List of token IDs with length equal to max_length
    """
    if len(token_ids) > max_length:
        # Truncate
        return token_ids[:max_length]
    else:
        # Pad
        return token_ids + [pad_token_id] * (max_length - len(token_ids))

def save_vocabulary(word2idx, output_dir='output'):
    """
    Save the vocabulary to a file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the vocabulary
    with open(os.path.join(output_dir, 'vocabulary.txt'), 'w') as f:
        for word, idx in sorted(word2idx.items(), key=lambda x: x[1]):
            f.write(f"{word}\t{idx}\n")
    
    print(f"Saved vocabulary to {os.path.join(output_dir, 'vocabulary.txt')}")

def load_vocabulary(filepath='output/vocabulary.txt'):
    """Load vocabulary from file."""
    word2idx = {}
    with open(filepath, 'r') as f:
        for line in f:
            word, idx = line.strip().split('\t')
            word2idx[word] = int(idx)
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

def save_token_ids(token_ids, output_dir='output'):
    """
    Save the token IDs to a file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy array
    np.save(os.path.join(output_dir, 'token_ids.npy'), np.array(token_ids))
    
    print(f"Saved token IDs to {os.path.join(output_dir, 'token_ids.npy')}")

def load_token_ids(filepath='output/token_ids.npy'):
    """Load token IDs from file."""
    return np.load(filepath)

def process_text_dataset(text, min_count=5, output_dir='output'):
    """Process text dataset end-to-end: tokenize, build vocab, convert to IDs, and save."""
    # Tokenize
    tokens = tokenize_text(text)
    
    # Build vocabulary
    word2idx, idx2word, word_counts = build_vocabulary(tokens, min_count)
    
    # Convert to token IDs
    token_ids = convert_to_token_ids(tokens, word2idx)
    
    # Save results
    save_vocabulary(word2idx, output_dir)
    save_token_ids(token_ids, output_dir)
    
    # Print statistics
    print("\nTokenization Statistics:")
    print(f"Total tokens: {len(tokens)}")
    print(f"Unique tokens: {len(set(tokens))}")
    print(f"Vocabulary size (with min_count={min_count}): {len(word2idx)}")
    print(f"Most common words: {word_counts.most_common(10)}")
    
    return tokens, word2idx, idx2word, token_ids

if __name__ == "__main__":
    process_text_dataset()
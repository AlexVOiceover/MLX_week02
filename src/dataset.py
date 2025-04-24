import torch
from torch.utils.data import Dataset
import json
from pathlib import Path
import random
import sys
import os
from dotenv import load_dotenv

# Add the project root to the path so we can import from src
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

from utils import tokenize_text, convert_to_token_ids, pad_or_truncate


class TripletDataset(Dataset):
    """
    Dataset for learning to rank using triplet loss.
    """

    def __init__(
        self,
        queries_file,
        passages_file,
        matches_file,
        word2idx,
        max_length=64,
        max_query_length=20,
    ):
        """
        Initialize the dataset.
        """
        # Load data
        with open(queries_file, "r") as f:
            self.queries = json.load(f)

        with open(passages_file, "r") as f:
            self.passages = json.load(f)

        with open(matches_file, "r") as f:
            self.matches = json.load(f)

        self.word2idx = word2idx
        self.max_length = max_length
        self.max_query_length = max_query_length

        # Check environment variable for negative sampling strategy
        use_cross_query = (
            os.getenv("USE_CROSS_QUERY_NEGATIVES", "False").lower() == "true"
        )
        negatives_per_positive = int(os.getenv("NEGATIVES_PER_POSITIVE", "6"))

        # Generate triplets based on strategy
        if use_cross_query:
            self.triplets = self._generate_cross_query_triplets(negatives_per_positive)
            print(
                f"Using CROSS-QUERY negative sampling strategy with {negatives_per_positive} negatives per positive"
            )
        else:
            self.triplets = self._generate_in_query_triplets()
            print("Using IN-QUERY negative sampling strategy")

        print(f"Generated {len(self.triplets)} triplets for training")

    def __len__(self):
        return len(self.triplets)

    def _generate_in_query_triplets(self):
        """Generate triplets using negatives from the same query's suggested documents."""
        triplets = []
        for query_id, match in self.matches.items():
            positive_id = match["selected"]
            for negative_id in match["suggested"]:
                if negative_id != positive_id:
                    triplets.append((query_id, positive_id, negative_id))
        return triplets

    def _generate_cross_query_triplets(self, negatives_per_positive):
        """Generate triplets using negatives from other queries."""
        from tqdm import tqdm
        
        print("Generating cross-query triplets...")
        triplets = []
        
        # Get all passage IDs at once - using a set for faster lookup
        all_passage_ids_set = set(self.passages.keys())
        print(f"Total unique passages: {len(all_passage_ids_set)}")
        
        # Create a sample pool of passages for negative sampling
        # (limiting to a reasonable size for performance)
        max_sample_pool = 100000
        if len(all_passage_ids_set) > max_sample_pool:
            print(f"Using a random sample of {max_sample_pool} passages for negative sampling")
            sample_pool = set(random.sample(list(all_passage_ids_set), max_sample_pool))
        else:
            sample_pool = all_passage_ids_set
            
        # Process each query with a progress bar
        print(f"Processing {len(self.matches)} queries...")
        for query_id, match in tqdm(self.matches.items()):
            # Only use queries that have a selected passage
            if "selected" not in match:
                continue
                
            positive_id = match["selected"]
            
            # Get the IDs to exclude (passages from this query)
            excluded_ids = set(match.get("suggested", []))
            
            # Sample negative passages from the pool
            # (excluding passages from this query)
            available_ids = list(sample_pool - excluded_ids)
            
            # If too few available IDs, skip this query
            if len(available_ids) < negatives_per_positive:
                continue
                
            # Sample negatives
            negative_ids = random.sample(available_ids, negatives_per_positive)
            
            # Create triplets
            for negative_id in negative_ids:
                triplets.append((query_id, positive_id, negative_id))
        
        print(f"Generated {len(triplets)} triplets")
        return triplets

    def _process_text(self, text, max_length):
        """
        Process text through tokenization, conversion to IDs, and padding.
        """
        # Tokenize the text
        tokens = tokenize_text(text)

        # Convert to token IDs
        token_ids = convert_to_token_ids(tokens, self.word2idx)

        # Pad or truncate
        token_ids = pad_or_truncate(token_ids, max_length)

        return torch.tensor(token_ids)

    def __getitem__(self, idx):
        # Get the triplet
        query_id, positive_id, negative_id = self.triplets[idx]

        # Get text
        query_text = self.queries[query_id]
        positive_text = self.passages[positive_id]
        negative_text = self.passages[negative_id]

        # Process text to tensors
        query_tensor = self._process_text(query_text, self.max_query_length)
        positive_tensor = self._process_text(positive_text, self.max_length)
        negative_tensor = self._process_text(negative_text, self.max_length)

        return query_tensor, positive_tensor, negative_tensor


if __name__ == "__main__":
    # Simple test
    from embeddings import load_pretrained_embedding

    # Get embedding vocabulary
    _, word2idx, idx2word = load_pretrained_embedding()

    # Load dataset
    data_dir = Path(__file__).parent.parent / "data" / "processed"
    dataset = TripletDataset(
        queries_file=data_dir / "queries.json",
        passages_file=data_dir / "passages.json",
        matches_file=data_dir / "matches.json",
        word2idx=word2idx,
    )

    print(f"Dataset has {len(dataset)} triplets")

    # Check a sample
    if len(dataset) > 0:
        # Get a sample triplet
        idx = 0  # First triplet
        query_tensor, positive_tensor, negative_tensor = dataset[idx]
        print(
            f"Shapes: Query {query_tensor.shape}, Positive {positive_tensor.shape}, Negative {negative_tensor.shape}"
        )

        # Show the original triplet
        query_id, positive_id, negative_id = dataset.triplets[idx]
        print("\nOriginal texts:")
        print(f"Query: {dataset.queries[query_id]}")
        print(
            f"Positive: {dataset.passages[positive_id][:100]}..."
            if len(dataset.passages[positive_id]) > 100
            else dataset.passages[positive_id]
        )
        print(
            f"Negative: {dataset.passages[negative_id][:100]}..."
            if len(dataset.passages[negative_id]) > 100
            else dataset.passages[negative_id]
        )

        # Show the tokenized version
        print("\nTokenized (first 10 tokens):")
        query_tokens = [
            idx2word.get(i.item(), "<UNK>") for i in query_tensor if i.item() != 0
        ][:10]
        positive_tokens = [
            idx2word.get(i.item(), "<UNK>") for i in positive_tensor if i.item() != 0
        ][:10]
        negative_tokens = [
            idx2word.get(i.item(), "<UNK>") for i in negative_tensor if i.item() != 0
        ][:10]

        print(f"Query tokens: {query_tokens}")
        print(f"Positive tokens: {positive_tokens}")
        print(f"Negative tokens: {negative_tokens}")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
from pathlib import Path
import wandb
from dotenv import load_dotenv
import time
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from embeddings import load_pretrained_embedding
from model import QueryTower, DocumentTower, triplet_loss
from dataset import TripletDataset

# Load environment variables
load_dotenv()

# Set up device for GPU training if available
# This creates a device that will be either GPU (cuda) or CPU depending on availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train():
    """
    Main training function for the ranking model.
    """
    print("Starting training...")

    # Define hyperparameters and training config
    config = {
        "learning_rate": 0.003,
        "batch_size": 1024,
        "epochs": 10,
        "triplet_margin": 0.2,
        "embedding_model": "Apples96/cbow_model_full",
        "optimizer": "Adam",
        # Add negative sampling configuration from environment
        "use_cross_query_negatives": os.getenv(
            "USE_CROSS_QUERY_NEGATIVES", "False"
        ).lower()
        == "true",
        "negatives_per_positive": int(os.getenv("NEGATIVES_PER_POSITIVE", "6")),
    }

    # Initialize Weights & Biases
    run = wandb.init(
        project="learn-to-search",
        config=config,
        name=f"ltr-model-{time.strftime('%Y%m%d-%H%M%S')}",
    )
    print(f"W&B initialized: {run.name}")

    # Step 1: Load pre-trained embedding model
    print("Loading pre-trained embedding model...")
    embedding_layer, word2idx, _ = load_pretrained_embedding()
    print(f"Loaded embedding model with vocabulary size: {len(word2idx)}")

    # Step 2: Create query and document towers
    print("Creating model towers...")
    query_tower = QueryTower(embedding_layer).to(device)  # Move model to GPU/CPU
    doc_tower = DocumentTower(embedding_layer).to(device)  # Move model to GPU/CPU
    print("Model towers created successfully.")

    # Step 3: Create dataset and dataloader
    print("Setting up dataset and dataloader...")
    data_dir = Path(__file__).parent.parent / "data" / "processed"

    # Log negative sampling strategy
    if config["use_cross_query_negatives"]:
        print(
            f"\nUsing CROSS-QUERY negative sampling with {config['negatives_per_positive']} negatives per positive"
        )
    else:
        print("\nUsing IN-QUERY negative sampling")

    # Create dataset
    dataset = TripletDataset(
        queries_file=data_dir / "queries.json",
        passages_file=data_dir / "passages.json",
        matches_file=data_dir / "matches.json",
        word2idx=word2idx,
    )

    # Create dataloader
    batch_size = config["batch_size"]
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    print(f"Dataset created with {len(dataset)} triplets")
    print(f"Dataloader created with batch size {batch_size}")

    # We'll log metrics during training instead of static information

    # Step 4: Set up optimizer
    print("Setting up optimizer...")

    # Collect only trainable parameters from both towers (exclude shared embedding)
    trainable_params = list(
        filter(lambda p: p.requires_grad, query_tower.parameters())
    ) + list(filter(lambda p: p.requires_grad, doc_tower.parameters()))

    # Create the optimizer with Adam
    optimizer = optim.Adam(trainable_params, lr=config["learning_rate"])

    # Count trainable parameters
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Optimizer created with learning rate {config['learning_rate']}")
    print(f"Total trainable parameters: {trainable_count}")

    # Log model architecture to W&B
    wandb.watch((query_tower, doc_tower), log="all")

    # Step 5: Training loop
    print("Starting training...")
    num_epochs = config["epochs"]

    # Initialize tracking variables
    prev_epoch_loss = float("inf")

    # Use tqdm for epoch tracking as well
    for epoch in tqdm(range(num_epochs), desc="Training progress", position=0):
        # Print GPU memory usage statistics at the beginning of each epoch
        if torch.cuda.is_available():
            print(
                f"CUDA Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
            )
            print(
                f"CUDA Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB"
            )
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")

        # Initialize metrics for this epoch
        epoch_loss = 0.0
        batch_count = 0

        # Process each batch
        for query_batch, pos_doc_batch, neg_doc_batch in tqdm(
            dataloader, desc=f"Epoch {epoch+1}", position=1, leave=False
        ):
            # Move data to device (GPU if available, otherwise CPU)
            query_batch = query_batch.to(device)
            pos_doc_batch = pos_doc_batch.to(device)
            neg_doc_batch = neg_doc_batch.to(device)

            # Forward pass
            query_embeddings = query_tower(query_batch)
            pos_doc_embeddings = doc_tower(pos_doc_batch)
            neg_doc_embeddings = doc_tower(neg_doc_batch)

            # Calculate loss
            batch_loss = triplet_loss(
                query_embeddings,
                pos_doc_embeddings,
                neg_doc_embeddings,
                margin=config["triplet_margin"],
            )

            # Backward pass
            optimizer.zero_grad()  # Zero the gradients
            batch_loss.backward()  # Calculate gradients
            optimizer.step()  # Update weights

            # Update metrics
            batch_loss_value = batch_loss.item()
            epoch_loss += batch_loss_value
            batch_count += 1

            # Log only batch loss with global step for x-axis
            global_step = batch_count + epoch * len(dataloader)
            wandb.log({"loss / batch": batch_loss_value}, step=global_step)

        # Print epoch statistics
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0

        # Calculate improvement from previous epoch
        loss_change = prev_epoch_loss - avg_loss
        improvement = (
            (loss_change / prev_epoch_loss * 100) if prev_epoch_loss > 0 else 0
        )

        # Print with improvement information
        if epoch > 0:
            if loss_change > 0:
                print(
                    f"  Average loss: {avg_loss:.4f} (improved by {improvement:.2f}%)"
                )
            else:
                print(
                    f"  Average loss: {avg_loss:.4f} (worsened by {-improvement:.2f}%)"
                )
        else:
            print(f"  Average loss: {avg_loss:.4f}")

        # Store current loss for next epoch comparison
        prev_epoch_loss = avg_loss

        # Log epoch-level metrics (average loss and improvement)
        # We use same step system so they appear at the right point in the timeline
        global_step = (epoch + 1) * len(dataloader)
        metrics = {"epoch_avg_loss": avg_loss}

        # Only log improvement metrics after first epoch
        if epoch > 0:
            metrics["loss_improvement"] = loss_change

        wandb.log(metrics, step=global_step)

    # Step 6: Save model
    model_dir = Path(__file__).parent.parent / "models" / "saved"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save the models
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    print("Saving models...")
    model_path = model_dir / f"ranking_model_{timestamp}.pt"
    print(f"Model timestamp: {timestamp}")

    # Create a state dictionary with both models
    state_dict = {
        "query_tower": query_tower.state_dict(),
        "doc_tower": doc_tower.state_dict(),
        "config": config,
        "epochs_trained": num_epochs,
    }

    # Save to file
    torch.save(state_dict, model_path)
    print(f"Model saved to {model_path}")

    # Create and log model as W&B Artifact
    artifact_name = f"ranking_model_{timestamp}"
    model_artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description="Dual-tower ranking model with query and document encoders",
        metadata=config,
    )
    print(f"Creating W&B artifact: {artifact_name}")

    # Add the model file to the artifact
    model_artifact.add_file(str(model_path))

    # Log the artifact to W&B
    wandb.log_artifact(model_artifact)

    print(f"Model saved as W&B artifact: {model_artifact.name}")
    print("Training complete!")

    # Close W&B run
    wandb.finish()


if __name__ == "__main__":
    train()

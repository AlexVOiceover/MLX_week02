#!/bin/bash
# This script builds a custom ChromaDB image with your data

# Step 1: Make sure ChromaDB is running and data is indexed
echo "Checking if ChromaDB container is running..."
if ! docker ps | grep -q chroma; then
  echo "Starting ChromaDB container..."
  docker-compose up -d
  echo "Waiting for ChromaDB to initialize..."
  sleep 5
fi

# Step 2: Check if data exists, if not ask to run indexer
if [ ! "$(ls -A ./data/index 2>/dev/null)" ]; then
  echo "Data directory is empty. Do you want to run the indexer? (y/n)"
  read -r answer
  if [ "$answer" = "y" ]; then
    echo "Running indexer..."
    export USE_REMOTE_CHROMA=True
    python src/indexer.py
  else
    echo "Skipping indexing. You'll need indexed data for the custom image."
  fi
fi

# Step 3: Build the custom image
echo "Building custom ChromaDB image..."
docker-compose -f docker-compose.custom.yml build

# Step 4: Tag the image for distribution
echo "Tagging image as chroma-with-data:latest"
docker tag mlx_week02-chroma:latest chroma-with-data:latest

echo "Your custom ChromaDB image is ready!"
echo "You can run it locally with: docker-compose -f docker-compose.custom.yml up -d"
echo "Or save it for distribution with: docker save chroma-with-data:latest | gzip > chroma-with-data.tar.gz"
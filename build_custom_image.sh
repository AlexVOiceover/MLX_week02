#!/bin/bash
# This script builds a custom ChromaDB image with your data and saves it to a file

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

# Step 5: Save the image to a file
echo "Saving image to chroma-with-data.tar.gz..."
docker save chroma-with-data:latest | gzip > chroma-with-data.tar.gz

echo "✅ Your custom ChromaDB image is ready!"
echo "📦 Image saved to: $(pwd)/chroma-with-data.tar.gz ($(du -h chroma-with-data.tar.gz | cut -f1) in size)"
echo ""
echo "To run it locally:"
echo "  docker-compose -f docker-compose.custom.yml up -d"
echo ""
echo "To deploy on another machine:"
echo "1. Copy chroma-with-data.tar.gz and docker-compose.custom.yml to the target machine"
echo "2. Run: docker load < chroma-with-data.tar.gz"
echo "3. Run: docker-compose -f docker-compose.custom.yml up -d"
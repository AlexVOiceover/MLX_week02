#!/bin/bash
# Simple script to start ChromaDB and the search app

# Set environment variables for remote ChromaDB
export USE_REMOTE_CHROMA=true
export CHROMA_HOST=localhost
export CHROMA_PORT=8000

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running. Please start Docker first."
  exit 1
fi

# Start ChromaDB container
echo "Starting ChromaDB container..."
docker-compose up -d

# Wait for ChromaDB to start
echo "Waiting for ChromaDB to start (5 seconds)..."
sleep 5

# Start the search app
echo "Starting search app..."
python src/launch_app.py

# When app is closed, stop containers
echo "Stopping containers..."
docker-compose down
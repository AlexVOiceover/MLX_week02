#!/bin/bash
# This script deploys a pre-built ChromaDB image with your data
# Run this on the target machine after copying chroma-with-data.tar.gz

# Check if chroma-with-data.tar.gz exists
if [ ! -f "chroma-with-data.tar.gz" ]; then
  echo "❌ Error: chroma-with-data.tar.gz not found!"
  echo "Please copy the image file to this directory first."
  exit 1
fi

# Check if docker-compose.custom.yml exists
if [ ! -f "docker-compose.custom.yml" ]; then
  echo "❌ Error: docker-compose.custom.yml not found!"
  echo "Creating a basic one for you..."
  cat > docker-compose.custom.yml << 'EOF'
services:
  chroma:
    image: chroma-with-data:latest
    ports:
      - "8000:8000"
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_PORT=8000
EOF
  echo "✅ Created docker-compose.custom.yml"
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "❌ Error: Docker is not running. Please start Docker first."
  exit 1
fi

# Load the image
echo "📥 Loading ChromaDB image (this may take a while)..."
docker load < chroma-with-data.tar.gz

# Start the container
echo "🚀 Starting ChromaDB container..."
docker-compose -f docker-compose.custom.yml up -d

# Check if container started successfully
if docker ps | grep -q chroma; then
  echo "✅ ChromaDB container is now running!"
  echo "🔌 API is available at: http://localhost:8000"
  echo ""
  echo "To use with the search app, make sure your .env file contains:"
  echo "USE_REMOTE_CHROMA=True"
  echo "CHROMA_HOST=localhost"
  echo "CHROMA_PORT=8000"
else
  echo "❌ Error: Container failed to start. Check 'docker logs' for details."
fi
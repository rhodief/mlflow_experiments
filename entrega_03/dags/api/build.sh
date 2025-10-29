#!/bin/bash

# Build script for Iris Classification API Docker image

echo "Building Iris Classification API Docker image..."

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Build the Docker image
docker build -t iris-classification-api:latest "$SCRIPT_DIR"

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully!"
    echo ""
    echo "Image: iris-classification-api:latest"
    echo ""
    echo "To run the container, execute: ./run.sh"
else
    echo "✗ Error building Docker image"
    exit 1
fi

#!/bin/bash

# Run script for Iris Classification API Docker container

echo "Starting Iris Classification API..."

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Path to assets directory (one level up from api folder)
ASSETS_DIR="$SCRIPT_DIR/../assets"

# Check if assets directory exists
if [ ! -d "$ASSETS_DIR" ]; then
    echo "✗ Error: Assets directory not found at $ASSETS_DIR"
    exit 1
fi

# Check if model files exist
if [ ! -f "$ASSETS_DIR/iris_random_forest.onnx" ]; then
    echo "✗ Error: ONNX model not found at $ASSETS_DIR/iris_random_forest.onnx"
    echo "Please run the Airflow DAG first to generate the model"
    exit 1
fi

if [ ! -f "$ASSETS_DIR/model_asset_metadata.json" ]; then
    echo "✗ Error: Model metadata not found at $ASSETS_DIR/model_asset_metadata.json"
    echo "Please run the Airflow DAG first to generate the metadata"
    exit 1
fi

# Stop and remove existing container if it exists
docker stop iris-api 2>/dev/null
docker rm iris-api 2>/dev/null

# Run the Docker container with privileged mode to allow ONNX runtime
docker run -d \
    --name iris-api \
    -p 8000:8000 \
    -v "$ASSETS_DIR:/app/assets:ro" \
    --restart unless-stopped \
    iris-classification-api:latest

if [ $? -eq 0 ]; then
    echo "✓ Container started successfully!"
    echo ""
    echo "API is running at: http://localhost:8000"
    echo "Swagger documentation: http://localhost:8000/docs"
    echo "ReDoc documentation: http://localhost:8000/redoc"
    echo ""
    echo "To view logs: docker logs -f iris-api"
    echo "To stop: docker stop iris-api"
    echo ""
    echo "Waiting for API to be ready..."
    sleep 3
    
    # Check if API is responding
    if curl -s http://localhost:8000/ > /dev/null; then
        echo "✓ API is ready and responding!"
    else
        echo "⚠ API may still be starting up. Check logs with: docker logs iris-api"
    fi
else
    echo "✗ Error starting Docker container"
    exit 1
fi

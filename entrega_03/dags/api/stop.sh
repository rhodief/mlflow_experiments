#!/bin/bash

# Stop script for Iris Classification API Docker container

echo "Stopping Iris Classification API..."

docker stop iris-api

if [ $? -eq 0 ]; then
    echo "✓ Container stopped successfully!"
    
    # Ask if user wants to remove the container
    read -p "Do you want to remove the container? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker rm iris-api
        echo "✓ Container removed"
    fi
else
    echo "✗ Error stopping container (it may not be running)"
    exit 1
fi

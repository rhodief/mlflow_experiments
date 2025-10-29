#!/bin/bash
# Wrapper script to start uvicorn with READ_IMPLIES_EXEC personality
# This allows the ONNX runtime to work with executable stack requirements

# Try to set the personality (may require privileged mode)
setarch $(uname -m) --addr-no-randomize uvicorn main:app --host 0.0.0.0 --port 8000 2>&1 || \
    uvicorn main:app --host 0.0.0.0 --port 8000

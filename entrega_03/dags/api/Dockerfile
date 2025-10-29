FROM python:3.12-slim

WORKDIR /app

# Install dependencies
RUN pip install -U pip && \
    pip install fastapi uvicorn numpy onnxruntime pydantic requests

# Copy application files
COPY main.py .

# Create assets directory for volume mount
RUN mkdir -p /app/assets

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Build: docker build -t iris-classification-api .
# Run: docker run -d -p 8000:8000 -v $(pwd)/../assets:/app/assets:ro --name iris-api iris-classification-api

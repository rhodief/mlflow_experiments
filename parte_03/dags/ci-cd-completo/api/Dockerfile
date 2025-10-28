FROM python:3.12-slim

RUN pip install -U numpy fastapi onnxruntime pydantic scikit-learn onnx pandas uvicorn

COPY ./artifacts/aula-16-10-25.onnx .
COPY ./artifacts/scaler.pkl .
COPY main.py .

CMD ["uvicorn", "main:APP", "--host", "0.0.0.0", "--port", "80"]

# Build da imagem: docker build -t api-inferencia .
# Execução do contêiner: docker run -it -p 5002:80 api-inferencia
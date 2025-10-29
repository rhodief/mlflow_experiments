from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import onnxruntime as ort
import numpy as np
import time
from datetime import datetime
import json
import os

# Initialize FastAPI app
app = FastAPI(
    title="API Classificação IRIS",
    description="API para classificação de flores Iris usando o modelo ONNX Random Forest",
    version="1.0.0"
)

# Global variables for metrics tracking
prediction_count = 0
total_latency = 0.0
predictions_history = []

# Load model metadata - support both Docker and host mode
MODEL_METADATA_PATH = os.getenv("MODEL_METADATA_PATH", "/app/assets/model_asset_metadata.json")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/assets/iris_random_forest.onnx")

# Load metadata
with open(MODEL_METADATA_PATH, 'r') as f:
    model_metadata = json.load(f)

# Feature names and target names
FEATURE_NAMES = model_metadata['features']
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

# Load ONNX model
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    print(f"✓ ONNX model loaded successfully from {MODEL_PATH}")
    print(f"  Model: {model_metadata['model_name']}")
    print(f"  Accuracy: {model_metadata['metrics']['accuracy']:.4f}")
except Exception as e:
    print(f"✗ Error loading ONNX model: {str(e)}")
    raise


# Pydantic models for request/response
class IrisFeatures(BaseModel):
    """Características de entrada para classificação de íris"""
    sepal_length: float = Field(..., description="Comprimento da sépala em cm", ge=0, le=10)
    sepal_width: float = Field(..., description="Largura da sépala em cm", ge=0, le=10)
    petal_length: float = Field(..., description="Comprimento da pétala em cm", ge=0, le=10)
    petal_width: float = Field(..., description="Largura da pétala em cm", ge=0, le=10)
    
    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """Modelo de resposta para previsões"""
    prediction: str = Field(..., description="Espécie de íris prevista")
    prediction_class: int = Field(..., description="Índice da classe prevista (0, 1 ou 2)")
    probabilities: Dict[str, float] = Field(..., description="Probabilidades de previsão para cada classe")
    duration_ms: float = Field(..., description="Duração da previsão em milissegundos")
    timestamp: str = Field(..., description="Data e hora da previsão")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "setosa",
                "prediction_class": 0,
                "probabilities": {
                    "setosa": 0.98,
                    "versicolor": 0.01,
                    "virginica": 0.01
                },
                "duration_ms": 2.5,
                "timestamp": "2025-10-29T14:30:00.123456"
            }
        }


class MetricsResponse(BaseModel):
    """Modelo de resposta para métricas"""
    total_predictions: int = Field(..., description="Número total de previsões realizadas")
    mean_latency_ms: float = Field(..., description="Latência média de previsão em milissegundos")
    model_info: Dict = Field(..., description="Metadados e desempenho do modelo")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_predictions": 100,
                "mean_latency_ms": 2.5,
                "model_info": {
                    "name": "iris_random_forest",
                    "type": "RandomForestClassifier",
                    "accuracy": 0.9,
                    "features": ["sepal length", "sepal width", "petal length", "petal width"]
                }
            }
        }


@app.get("/", tags=["Saúde"])
async def root():
    """
    Endpoint raiz - Verificação de saúde
    """
    return {
        "status": "saudável",
        "message": "API de Classificação Iris está em execução",
        "model": model_metadata['model_name'],
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Previsão"])
async def predict(features: IrisFeatures):
    """
    Fazer uma previsão de espécie de íris com base nas características de entrada.
    
    **Parâmetros:**
    - **sepal_length**: Comprimento da sépala em centímetros (0-10)
    - **sepal_width**: Largura da sépala em centímetros (0-10)
    - **petal_length**: Comprimento da pétala em centímetros (0-10)
    - **petal_width**: Largura da pétala em centímetros (0-10)
    
    **Retorna:**
    - **prediction**: Nome da espécie de íris prevista
    - **prediction_class**: Índice da classe (0=setosa, 1=versicolor, 2=virginica)
    - **probabilities**: Probabilidade de previsão para cada classe
    - **duration_ms**: Tempo de execução da previsão em milissegundos
    - **timestamp**: Quando a previsão foi feita
    """
    global prediction_count, total_latency
    
    try:
        # Start timing
        start_time = time.time()
        
        # Prepare input data
        input_data = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]], dtype=np.float32)
        
        # Get input name from ONNX model
        input_name = ort_session.get_inputs()[0].name
        
        # Make prediction
        outputs = ort_session.run(None, {input_name: input_data})
        
        # Extract prediction and probabilities
        prediction_class = int(outputs[0][0])
        probabilities = outputs[1][0]  # Get probabilities from second output
        
        # End timing
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Update metrics
        prediction_count += 1
        total_latency += duration_ms
        
        # Get predicted species name
        prediction_name = TARGET_NAMES[prediction_class]
        
        # Create probabilities dictionary
        prob_dict = {
            TARGET_NAMES[i]: float(probabilities[i]) 
            for i in range(len(TARGET_NAMES))
        }
        
        # Create response
        response = {
            "prediction": prediction_name,
            "prediction_class": prediction_class,
            "probabilities": prob_dict,
            "duration_ms": round(duration_ms, 3),
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Métricas"])
async def get_metrics():
    """
    Obter métricas da API incluindo total de previsões e latência média.
    
    **Retorna:**
    - **total_predictions**: Número total de previsões feitas desde que a API iniciou
    - **mean_latency_ms**: Tempo médio por previsão em milissegundos
    - **model_info**: Informações sobre o modelo carregado incluindo precisão e características
    """
    mean_latency = total_latency / prediction_count if prediction_count > 0 else 0.0
    
    return {
        "total_predictions": prediction_count,
        "mean_latency_ms": round(mean_latency, 3),
        "model_info": {
            "name": model_metadata['model_name'],
            "type": model_metadata['model_type'],
            "format": model_metadata['format'],
            "accuracy": model_metadata['metrics']['accuracy'],
            "precision": model_metadata['metrics']['precision'],
            "recall": model_metadata['metrics']['recall'],
            "f1_score": model_metadata['metrics']['f1_score'],
            "features": model_metadata['features'],
            "n_features": model_metadata['n_features'],
            "created_at": model_metadata['created_at']
        }
    }


@app.get("/model-info", tags=["Modelo"])
async def get_model_info():
    """
    Obter informações detalhadas sobre o modelo carregado.
    
    **Retorna:**
    Metadados completos do modelo incluindo métricas de treinamento e informações de características.
    """
    return model_metadata


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

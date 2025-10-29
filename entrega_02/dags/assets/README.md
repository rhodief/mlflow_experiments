# Relatório II - ML Pipeline Assets

Este diretório contém os artefatos gerados pelo pipeline de Machine Learning do DAG `relatorio-II`.

## Arquivos Gerados

### 1. `iris_data.pkl`
- Dataset Iris carregado do scikit-learn
- Contém features (X), targets (y) e nomes das classes

### 2. `iris_preprocessed.pkl`
- Dados preprocessados e divididos em treino/teste
- Features normalizadas com StandardScaler
- Train/test split: 80/20

### 3. `random_forest_model.pkl`
- Modelo Random Forest treinado
- Inclui scaler e metadados

### 4. `iris_random_forest.onnx`
- Modelo convertido para formato ONNX
- Pronto para deployment em produção
- Compatível com diversos frameworks de inferência

### 5. `model_asset_metadata.json`
- Metadados completos do modelo
- Métricas de avaliação
- Referências ao MLflow

## Pipeline Overview

```
ingestao_dados → pre_processamento → treinamento → teste → empacotamento → registro
```

### Etapas:

1. **Ingestão de Dados**: Carrega dataset Iris e cria experimento MLflow
2. **Pré-processamento**: Split e normalização dos dados
3. **Treinamento**: Random Forest com registro no MLflow
4. **Teste**: Avaliação com accuracy, precision, recall, f1-score
5. **Empacotamento**: Conversão para ONNX
6. **Registro**: Registro no MLflow Model Registry

## Acessar Artefatos

- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080
- **Assets locais**: `/opt/airflow/dags/assets/`

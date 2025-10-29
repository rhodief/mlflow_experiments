# 🚀 Quick Start - Deploy ML Pipeline with Docker

## TL;DR

```bash
# 1. Start the cluster
cd /home/rhodie/dev/lab/mlflow/entrega_03
./init.cluster.sh

# 2. Configure Docker Hub credentials (via UI at http://localhost:8080)
# Admin → Variables:
#   - docker_registry_username: your_username
#   - docker_registry_password: your_password

# 3. Run the DAG "relatorio-II" from Airflow UI

# 4. Test the API after deployment
curl http://localhost:8884/metrics
```

## What Was Extended?

The `dag_entrega_3.py` now includes a complete CI/CD pipeline:

### Before (Original)
```
Ingestão → Pré-processamento → Treinamento → Teste → Empacotamento → Registro
```

### After (Extended) ✅
```
Ingestão → Pré-processamento → Treinamento → Teste → Empacotamento → Registro
    ↓
Build Docker Image → Push to Docker Hub → Stop Old Container → Deploy New Container → Health Check
```

## New Tasks Added

1. **build_imagem** - Builds Docker image from `dags/api/Dockerfile`
2. **push_imagem** - Pushes image to Docker Hub
3. **verificar_container_executando** - Stops running container if exists
4. **deploy** - Starts new container with model mounted
5. **verificar_container_saude** - Tests all API endpoints

## API Endpoints

Once deployed, test at `http://localhost:8884`:

### GET /
Health check

### GET /metrics
Model metrics and API stats

### POST /predict
Make predictions
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## Parameters

Configure via Airflow UI when triggering:

```json
{
  "tag_imagem": "your_username/iris-api:latest",
  "nome_container": "iris-api",
  "porta_api": 8884
}
```

## URLs

- **Airflow UI**: http://localhost:8080 (airflow / airflow)
- **MLflow UI**: http://localhost:5000
- **API (after deploy)**: http://localhost:8884
- **Jupyter**: http://localhost:8848 (token: mlflow)

## Files

- `dag_entrega_3.py` - Extended DAG with Docker deployment
- `dags/api/Dockerfile` - API Docker image
- `dags/api/main.py` - FastAPI application
- `dags/assets/` - Generated artifacts (ONNX model, metadata)
- `test_api.py` - Automated API testing script

## Testing

After the DAG completes:

```bash
# Option 1: Use the test script
python3 test_api.py

# Option 2: Manual testing
curl http://localhost:8884/
curl http://localhost:8884/metrics
curl -X POST http://localhost:8884/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

## Troubleshooting

### Docker permission error
✅ **Already fixed!** The `group_add` in docker-compose.yml allows Airflow to access Docker.

### Push to Docker Hub fails
Configure credentials in Airflow Variables (Admin → Variables)

### Container already exists
The DAG automatically stops old containers before deploying

### API not responding
Wait ~5-10 seconds after deployment for the API to initialize

## Documentation

- `DEPLOYMENT_GUIDE.md` - Complete deployment documentation
- `CHANGES_SUMMARY.md` - Detailed changes to the DAG
- `FIX_DOCKER_PERMISSION.md` - Docker permission fix explanation
- `RESOLUTION_SUMMARY.md` - Problem resolution summary
- `QUICK_START.md` - This file

## Success Checklist

- [x] Docker permission issue resolved
- [x] Extended DAG with build/push/deploy tasks  
- [x] Health check implementation
- [x] API container deployment
- [x] Documentation created
- [x] Test script provided

✅ **Ready to deploy!**

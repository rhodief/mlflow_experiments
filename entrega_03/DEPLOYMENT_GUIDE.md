# Guia de Deploy - DAG Entrega 3

## Visão Geral

O DAG `relatorio-II` foi estendido para incluir as etapas completas de CI/CD, incluindo:

1. **Pipeline de ML**: Ingestão → Pré-processamento → Treinamento → Teste → Empacotamento → Registro
2. **Entrega (Build & Push)**: Construção da imagem Docker e push para Docker Hub
3. **Deploy**: Deploy do container e verificação de saúde

## Arquitetura do Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    ML PIPELINE                               │
├─────────────────────────────────────────────────────────────┤
│ ingestao_dados → pre_processamento → treinamento           │
│      ↓                                                       │
│   teste → empacotamento → registro                          │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                  DOCKER DEPLOYMENT                           │
├─────────────────────────────────────────────────────────────┤
│ build_imagem → push_imagem → verificar_container_executando│
│      ↓                                                       │
│   deploy → verificar_container_saude                        │
└─────────────────────────────────────────────────────────────┘
```

## Novas Etapas Adicionadas

### 1. build_imagem
Constrói a imagem Docker da API a partir do Dockerfile em `/opt/airflow/dags/api/`.

**Funcionalidade:**
- Constrói imagem Docker com o FastAPI e modelo ONNX
- Usa network mode 'host' para acesso ao registry
- Tag configurável via parâmetros do DAG

### 2. push_imagem
Envia a imagem construída para o Docker Hub.

**Funcionalidade:**
- Autentica no Docker Hub usando credenciais do Airflow Variables
- Faz push da imagem com a tag especificada
- Verifica erros durante o push

### 3. verificar_container_executando
Verifica se já existe um container com o mesmo nome em execução e o remove.

**Funcionalidade:**
- Lista containers pelo nome
- Remove container existente (force=True)
- Prepara ambiente para novo deploy

### 4. deploy
Executa o container Docker com a API.

**Funcionalidade:**
- Inicia container com a imagem do Docker Hub
- Monta volume com os artifacts (modelo ONNX e metadata)
- Expõe porta configurável (padrão: 8884)
- Configura variáveis de ambiente

### 5. verificar_container_saude
Verifica se o container está funcionando corretamente.

**Funcionalidade:**
- Verifica se container está running
- Testa endpoint raiz (`/`)
- Testa endpoint de métricas (`/metrics`)
- Testa endpoint de predição (`/predict`) com dados de exemplo
- Retorna resultados detalhados dos testes

## Parâmetros do DAG

O DAG aceita os seguintes parâmetros configuráveis:

```python
params = {
    'tag_imagem': 'iris-classification-api:latest',  # Tag da imagem Docker
    'nome_container': 'iris-api',                     # Nome do container
    'porta_api': 8884                                 # Porta para expor a API
}
```

### Como Configurar os Parâmetros

Ao executar o DAG manualmente na UI do Airflow, você pode modificar:

- **tag_imagem**: Nome e tag da imagem (ex: `seunome/iris-api:v1.0`)
- **nome_container**: Nome do container (ex: `iris-api-prod`)
- **porta_api**: Porta do host (ex: `8080`, `8884`, etc.)

## Configuração Necessária

### 1. Variáveis do Airflow

Configure as seguintes variáveis no Airflow UI (Admin → Variables):

```bash
docker_registry_username = seu_usuario_dockerhub
docker_registry_password = sua_senha_dockerhub
```

### 2. Conexão Docker

Certifique-se de que a conexão Docker está configurada:

- **Conn Id**: `docker`
- **Conn Type**: `Docker`
- **Host**: `unix://var/run/docker.sock` (ou TCP se necessário)

### 3. Estrutura de Diretórios

```
entrega_03/
├── dags/
│   ├── dag_entrega_3.py          # DAG principal
│   ├── api/
│   │   ├── Dockerfile            # Dockerfile da API
│   │   └── main.py               # Código FastAPI
│   └── assets/                   # Artifacts gerados
│       ├── iris_random_forest.onnx
│       ├── model_asset_metadata.json
│       └── (outros arquivos pickle)
```

## Endpoints da API

Após o deploy bem-sucedido, a API estará disponível em `http://localhost:8884` (ou porta configurada):

### GET /
Verificação de saúde básica.

**Resposta:**
```json
{
  "status": "saudável",
  "message": "API de Classificação Iris está em execução",
  "model": "iris_random_forest",
  "version": "1.0.0"
}
```

### GET /metrics
Retorna métricas da API.

**Resposta:**
```json
{
  "total_predictions": 0,
  "mean_latency_ms": 0.0,
  "model_info": {
    "name": "iris_random_forest",
    "type": "RandomForestClassifier",
    "accuracy": 0.9667,
    "precision": 0.9667,
    "recall": 0.9667,
    "f1_score": 0.9667,
    "features": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"],
    "n_features": 4
  }
}
```

### POST /predict
Faz uma predição de espécie de íris.

**Request Body:**
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

**Resposta:**
```json
{
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
```

## Como Executar

### 1. Via Airflow UI

1. Acesse o Airflow UI
2. Encontre o DAG `relatorio-II`
3. Clique em "Trigger DAG"
4. (Opcional) Configure os parâmetros
5. Confirme a execução

### 2. Via CLI

```bash
# Trigger o DAG com parâmetros padrão
airflow dags trigger relatorio-II

# Trigger com parâmetros customizados
airflow dags trigger relatorio-II \
  --conf '{"tag_imagem": "seunome/iris-api:v1.0", "nome_container": "iris-api", "porta_api": 8080}'
```

## Testando a API Manualmente

Após o deploy, você pode testar a API usando curl:

```bash
# Teste de saúde
curl http://localhost:8884/

# Obter métricas
curl http://localhost:8884/metrics

# Fazer uma predição
curl -X POST http://localhost:8884/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

## Verificação de Saúde Automática

A task `verificar_container_saude` realiza os seguintes testes automaticamente:

1. ✓ Verifica se o container está running
2. ✓ Testa endpoint raiz (`/`)
3. ✓ Testa endpoint de métricas (`/metrics`)
4. ✓ Testa predição com dados de exemplo (`/predict`)

Se algum teste falhar, a task levanta uma exceção e o DAG é marcado como falhado.

## Troubleshooting

### Container não inicia
- Verifique os logs: `docker logs iris-api`
- Certifique-se de que a porta não está em uso
- Verifique se os arquivos de artifacts existem

### Push para Docker Hub falha
- Verifique as credenciais nas Variables do Airflow
- Certifique-se de que o usuário tem permissão de push
- Verifique a conectividade com Docker Hub

### API não responde
- Aguarde alguns segundos após o deploy
- Verifique os logs do container
- Teste se a porta está acessível: `curl http://localhost:8884/`

### Erro "Model file not found"
- Verifique se o volume está montado corretamente
- Certifique-se de que os arquivos existem em `/opt/airflow/dags/assets/`

## Diferenças em Relação ao dag_ci_cd.py

O `dag_entrega_3.py` foi adaptado do `dag_ci_cd.py` com as seguintes diferenças:

1. **Dataset**: Usa Iris ao invés de Student Stress
2. **Modelo**: Random Forest ao invés de Decision Tree
3. **Estrutura**: Usa PythonOperator ao invés de @task decorators
4. **Porta**: Porta padrão 8884 ao invés de 8883
5. **Volume Mount**: Monta `/opt/airflow/dags/assets` ao invés de artifacts
6. **Health Check**: Implementação mais robusta com testes de todos os endpoints

## Próximos Passos

- Adicionar testes de carga
- Implementar rollback automático se health check falhar
- Adicionar suporte para múltiplos ambientes (staging, production)
- Implementar versionamento semântico das imagens
- Adicionar monitoramento contínuo do container

# MLflow Experiments - Explicação Completa

## 🎯 O que é um Experimento no MLflow?

Um **Experimento** no MLflow é um **container lógico** que agrupa múltiplas execuções (runs) relacionadas a um mesmo projeto ou caso de uso de ML.

### Analogia:
```
Experimento = Projeto de ML
Run = Uma execução específica do pipeline
```

## 📊 Estrutura Hierárquica do MLflow

```
┌─────────────────────────────────────────────────────────────┐
│                    MLflow Tracking Server                    │
│                    http://mlflow:5000                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
   Experiment 1                         Experiment 2
   "relatorio-II-iris-pipeline"        "outro-projeto"
        │                                     │
        ├─ Run 1 (DAG execution #1)          ├─ Run 1
        │  ├─ Parameters                     │  └─ ...
        │  ├─ Metrics                        │
        │  ├─ Artifacts                      └─ Run 2
        │  └─ Tags                              └─ ...
        │
        ├─ Run 2 (DAG execution #2)
        │  ├─ Parameters
        │  ├─ Metrics
        │  ├─ Artifacts
        │  └─ Tags
        │
        └─ Run 3 (DAG execution #3)
           └─ ...
```

## ✅ Seu Setup Atual: CORRETO!

### Você tem:

**1 Experimento**: `relatorio-II-iris-pipeline`
- Criado na task `ingestao_dados`
- Compartilhado por TODAS as tasks do DAG

**1 Run por execução do DAG**:
- Criado na task `treinamento` com `mlflow.start_run()`
- O `run_id` é passado para as tasks seguintes via XCom
- Todas as tasks logam no **mesmo run**

### Fluxo no seu DAG:

```python
# Task 1: ingestao_dados
mlflow.set_experiment("relatorio-II-iris-pipeline")  # ← Cria/usa experimento

# Task 3: treinamento
with mlflow.start_run(run_name="random_forest_training") as run:  # ← Cria RUN
    mlflow.log_params(...)      # Loga parâmetros
    mlflow.sklearn.log_model()  # Loga modelo
    run_id = run.info.run_id    # Salva run_id

# Task 4: teste
with mlflow.start_run(run_id=run_id):  # ← USA O MESMO RUN
    mlflow.log_metric("accuracy", ...)
    mlflow.log_metric("precision", ...)
    # ... mais métricas

# Task 5: empacotamento
with mlflow.start_run(run_id=run_id):  # ← USA O MESMO RUN
    mlflow.log_artifact(onnx_path)
```

## 📈 O que acontece em cada execução do DAG?

### Primeira execução:
```
Experimento: relatorio-II-iris-pipeline
└── Run 1
    ├── Parameters: n_estimators=100, max_depth=10
    ├── Metrics: accuracy=0.9667, precision=0.9667, ...
    ├── Artifacts: modelo sklearn, ONNX, feature_importance
    └── Tags: run_name="random_forest_training"
```

### Segunda execução (se você executar de novo):
```
Experimento: relatorio-II-iris-pipeline
├── Run 1 (execução anterior)
└── Run 2 (nova execução)
    ├── Parameters: n_estimators=100, max_depth=10
    ├── Metrics: accuracy=0.9667, precision=0.9667, ...
    └── Artifacts: modelo sklearn, ONNX, feature_importance
```

## 🔍 Verificando no MLflow UI

### 1. Acesse: http://localhost:5000

### 2. Veja o Experimento:
- Clique em "Experiments" no menu lateral
- Você verá: `relatorio-II-iris-pipeline`
- Ao lado do nome: **1 run** (ou mais se executou múltiplas vezes)

### 3. Clique no Experimento:
Você verá uma **tabela com todos os runs**:

| Run Name | Created | Metrics | Parameters |
|----------|---------|---------|------------|
| random_forest_training | 2025-10-29 | acc: 0.9667 | n_est: 100 |

### 4. Clique em um Run:
Verá todos os detalhes:
- **Parameters**: n_estimators, max_depth, random_state
- **Metrics**: accuracy, precision, recall, f1_score
- **Artifacts**: 
  - model/ (sklearn model)
  - onnx_model/ (ONNX model)
  - feature_importance.json
  - confusion_matrix.json
  - classification_report.txt

## ❓ Quando ter múltiplos Experimentos?

Você criaria **experimentos separados** para:

### Exemplo 1: Diferentes Datasets
```python
# Experimento 1: Iris
mlflow.set_experiment("iris-classification")

# Experimento 2: Wine
mlflow.set_experiment("wine-classification")

# Experimento 3: Digits
mlflow.set_experiment("digits-classification")
```

### Exemplo 2: Diferentes Objetivos
```python
# Experimento 1: Exploração inicial
mlflow.set_experiment("iris-exploration")

# Experimento 2: Otimização de hiperparâmetros
mlflow.set_experiment("iris-hyperparameter-tuning")

# Experimento 3: Produção
mlflow.set_experiment("iris-production")
```

### Exemplo 3: Diferentes Algoritmos (se fossem projetos separados)
```python
# Experimento 1: Random Forest
mlflow.set_experiment("iris-random-forest")

# Experimento 2: SVM
mlflow.set_experiment("iris-svm")

# Experimento 3: Neural Network
mlflow.set_experiment("iris-neural-network")
```

## ❌ Você NÃO precisa de experimentos separados para cada task

**Errado** (não faça isso):
```python
# Task 1
mlflow.set_experiment("relatorio-II-ingestao")

# Task 3
mlflow.set_experiment("relatorio-II-treinamento")  # ❌ Separa tudo

# Task 4
mlflow.set_experiment("relatorio-II-teste")  # ❌ Perde a conexão
```

**Certo** (o que você já tem):
```python
# Task 1: Define o experimento UMA vez
mlflow.set_experiment("relatorio-II-iris-pipeline")

# Task 3: Cria um RUN dentro do experimento
with mlflow.start_run() as run:
    # Treina e loga
    pass

# Task 4: USA O MESMO RUN
with mlflow.start_run(run_id=run_id):
    # Loga métricas no mesmo run
    pass
```

## 🎓 Conceitos Importantes

### Experiment (Experimento)
- **O quê**: Container de runs relacionados
- **Quando criar**: Para cada projeto/caso de uso distinto
- **Quantos você tem**: 1 (`relatorio-II-iris-pipeline`) ✅
- **Quantos você deveria ter**: 1 ✅

### Run (Execução)
- **O quê**: Uma execução específica do seu código ML
- **Quando criar**: Cada vez que você treina um modelo
- **Quantos você tem**: 1 por execução do DAG
- **Quantos você deveria ter**: Quantas vezes executar o DAG

### Parameters (Parâmetros)
- **O quê**: Configurações do modelo (ex: n_estimators=100)
- **Onde**: Dentro de cada run
- **Quando logar**: Durante o treinamento

### Metrics (Métricas)
- **O quê**: Resultados de avaliação (ex: accuracy=0.9667)
- **Onde**: Dentro de cada run
- **Quando logar**: Durante o teste/avaliação

### Artifacts (Artefatos)
- **O quê**: Arquivos gerados (modelos, gráficos, etc.)
- **Onde**: Dentro de cada run
- **Quando logar**: Durante treinamento, empacotamento, etc.

## 📊 Visualização do seu Pipeline

```
╔═══════════════════════════════════════════════════════════════╗
║           Experimento: relatorio-II-iris-pipeline             ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Run 1: random_forest_training                                ║
║  ├─ Created by: task "treinamento"                           ║
║  ├─ Run ID: abc123...                                        ║
║  │                                                            ║
║  ├─ Parameters (logged by "treinamento")                     ║
║  │   ├─ n_estimators: 100                                    ║
║  │   ├─ max_depth: 10                                        ║
║  │   └─ random_state: 42                                     ║
║  │                                                            ║
║  ├─ Metrics (logged by "teste")                              ║
║  │   ├─ accuracy: 0.9667                                     ║
║  │   ├─ precision: 0.9667                                    ║
║  │   ├─ recall: 0.9667                                       ║
║  │   └─ f1_score: 0.9667                                     ║
║  │                                                            ║
║  └─ Artifacts                                                 ║
║      ├─ model/ (logged by "treinamento")                     ║
║      │   └─ sklearn model                                    ║
║      ├─ onnx_model/ (logged by "empacotamento")              ║
║      │   └─ iris_random_forest.onnx                          ║
║      ├─ feature_importance.json (logged by "treinamento")    ║
║      ├─ confusion_matrix.json (logged by "teste")            ║
║      └─ classification_report.txt (logged by "teste")        ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

## 🔍 Comandos para Verificar

### Ver todos os experimentos
```bash
docker compose exec airflow-apiserver python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f'ID: {exp.experiment_id}, Nome: {exp.name}')
"
```

### Ver runs de um experimento
```bash
docker compose exec airflow-apiserver python -c "
import mlflow
mlflow.set_tracking_uri('http://mlflow:5000')
runs = mlflow.search_runs(experiment_names=['relatorio-II-iris-pipeline'])
print(runs[['run_id', 'metrics.accuracy', 'params.n_estimators']])
"
```

## ✅ Resumo: Está Tudo Certo!

| Item | Seu Setup | Status |
|------|-----------|--------|
| Número de experimentos | 1 | ✅ Correto |
| Nome do experimento | relatorio-II-iris-pipeline | ✅ Correto |
| Runs por execução do DAG | 1 | ✅ Correto |
| Compartilhamento do run_id | Sim, via XCom | ✅ Correto |
| Logs no mesmo run | Sim | ✅ Correto |

## 🎯 Conclusão

**SIM, ter 1 experimento está CORRETO!**

Você verá:
- **1 Experimento**: `relatorio-II-iris-pipeline`
- **N Runs**: Um para cada vez que executar o DAG

Cada run conterá:
- Parâmetros do modelo
- Métricas de avaliação
- Artefatos (modelos, ONNX, reports)

Isso é a **best practice** para um pipeline de ML integrado! 🎉

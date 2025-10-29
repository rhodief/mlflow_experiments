# Tags Information - DAG relatorio-II

## ✅ Tags Confirmados

O DAG `relatorio-II` possui as seguintes tags configuradas e **registradas no sistema**:

1. 🔵 **mlflow** - Indica integração com MLflow
2. 🌸 **iris** - Dataset utilizado (Iris)
3. 🌲 **random-forest** - Algoritmo de ML utilizado
4. 📦 **onnx** - Formato de exportação do modelo

## 🔍 Verificação

As tags foram verificadas e confirmadas através do comando:

```bash
docker compose exec airflow-dag-processor airflow dags details relatorio-II
```

**Resultado:**
```
tags: {'name': 'onnx', 'dag_id': 'relatorio-II'},
      {'name': 'random-forest', 'dag_id': 'relatorio-II'},
      {'name': 'iris', 'dag_id': 'relatorio-II'},
      {'name': 'mlflow', 'dag_id': 'relatorio-II'}
```

## 🖥️ Visualização no Airflow UI

### Se as tags não aparecem na UI:

1. **Atualize o navegador** (Ctrl + F5 ou Cmd + Shift + R)
   - O cache do browser pode estar ocultando as tags

2. **Verifique se o DAG está despausado**
   ```bash
   docker compose exec airflow-apiserver airflow dags unpause relatorio-II
   ```

3. **Aguarde alguns segundos**
   - O Airflow UI pode demorar um pouco para sincronizar

4. **Limpe o cache do navegador**
   - Settings → Privacy → Clear browsing data
   - Ou use modo incognito/privado

5. **Reinicie o apiserver se necessário**
   ```bash
   docker compose restart airflow-apiserver
   ```

## 📍 Onde Encontrar as Tags na UI

No Airflow 3 UI, as tags podem aparecer em diferentes lugares:

1. **DAGs View (Lista principal)**
   - Abaixo ou ao lado do nome do DAG
   - Como pequenos badges coloridos

2. **DAG Details Page**
   - Na página de detalhes do DAG
   - Seção "Tags" ou "Metadata"

3. **Grid View**
   - No topo da página, próximo ao nome do DAG

## 🔧 Comandos Úteis

### Listar todos os DAGs com tags
```bash
docker compose exec airflow-dag-processor airflow dags list
```

### Ver detalhes específicos do DAG
```bash
docker compose exec airflow-dag-processor airflow dags details relatorio-II
```

### Listar apenas os DAGs com tag específica
```bash
# No Python dentro do container
docker compose exec airflow-apiserver python -c "
from airflow.models import DagBag, DagTag
from airflow import settings
session = settings.Session()
tags = session.query(DagTag).filter(DagTag.name=='mlflow').all()
for tag in tags:
    print(f'{tag.dag_id}: {tag.name}')
"
```

## 📝 Código das Tags no DAG

No arquivo `dag_entrega_2.py`, linha 455:

```python
dag = DAG(
    "relatorio-II",
    schedule=None,
    description="Pipeline completo de ML com Iris dataset usando MLflow e ONNX",
    default_args={
        'depends_on_past': False,
        'owner': 'airflow'
    },
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlflow', 'iris', 'random-forest', 'onnx']  # ← Tags aqui
)
```

## 🎯 Status Atual

- ✅ DAG registrado no Airflow
- ✅ Tags configuradas no código
- ✅ Tags armazenadas no banco de dados
- ✅ DAG despausado (ativo)
- ✅ Pronto para execução

## 🌐 Acessar Airflow UI

```
URL: http://localhost:8080
Username: airflow
Password: airflow
```

Após fazer login, procure por `relatorio-II` na lista de DAGs.

## ⚠️ Nota sobre Airflow 3

O Airflow 3 pode ter mudanças na forma como as tags são exibidas na interface. 
Se você não vê as tags visualmente, elas AINDA ESTÃO LÁ no sistema e funcionam 
para filtros e buscas.

Você pode usar a busca por tags na barra de filtros da UI.

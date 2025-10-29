from airflow.sdk import DAG, task_group
from airflow.providers.standard.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from docker import DockerClient as DockerClientAPI
from datetime import datetime
import json
import os


class DockerClient(BaseHook):
    '''Docker client that can work with or without Airflow connections'''
    def __init__(self, docker_conn_id: str = None):
        super().__init__()
        self._docker_conn_id = docker_conn_id

    def from_env(self, *args, **kwargs):
        # Try to get connection from Airflow, fallback to docker socket
        try:
            if self._docker_conn_id:
                conn = self.get_connection(self._docker_conn_id)
                host = conn.host
                port = conn.port

                if 'tcp://' not in host:
                    host = "tcp://" + host
                
                return DockerClientAPI(base_url=f"{host}:{port}", *args, **kwargs)
        except Exception as e:
            print(f"Could not get Docker connection '{self._docker_conn_id}': {e}")
            print("Falling back to unix socket")
        
        # Fallback to unix socket
        return DockerClientAPI(base_url="unix://var/run/docker.sock", *args, **kwargs)


MLFLOW_CONNECTION_ID = "mlflow"
DOCKER_CONNECTION_ID = "docker"
DOCKERFILE_PATH = "/opt/airflow/dags/api/"
ARTIFACTS_PATH = "/opt/airflow/dags/assets/"

# Host path for Docker volume mounts (when Airflow runs in container with Docker socket)
# Detect from environment variable or construct from current file path
def get_host_artifacts_path():
    """
    Detecta o caminho correto dos assets tanto dentro quanto fora do container.
    Quando dentro do Airflow container, retorna o caminho do host para volume mounts.
    """
    # Detecta se está rodando dentro de um container
    in_container = os.path.exists('/.dockerenv')
    
    if in_container:
        # Dentro do container Airflow, precisa retornar o caminho do HOST
        # HOST_PWD é passado via docker-compose.yml e contém o caminho do host
        host_pwd = os.getenv('HOST_PWD')
        if host_pwd:
            return os.path.join(host_pwd, 'dags', 'assets')
        
        # Fallback: tenta AIRFLOW_PROJ_DIR
        airflow_proj_dir = os.getenv('AIRFLOW_PROJ_DIR')
        if airflow_proj_dir and airflow_proj_dir != '.':
            return os.path.join(airflow_proj_dir, 'dags', 'assets')
        
        # Último fallback: usa PWD
        pwd = os.getenv('PWD')
        if pwd:
            return os.path.join(pwd, 'dags', 'assets')
        
        # Se tudo falhar, usa o diretório atual com warning
        print("WARNING: Could not determine host path, using container path as fallback")
        return os.path.join(os.getcwd(), 'dags', 'assets')
    else:
        # Fora do container, usa __file__ para construir o caminho
        dag_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(dag_dir, 'assets')

ARTIFACTS_HOST_PATH = get_host_artifacts_path()
print(f"DEBUG: ARTIFACTS_HOST_PATH = {ARTIFACTS_HOST_PATH}")
MLFLOW_ADDRESS = "http://mlflow:5000/"


def ingestao_dados(**context):
    """
    Load the iris dataset from scikit-learn and setup a new experiment in MLFlow
    """
    import mlflow
    from sklearn.datasets import load_iris
    import pandas as pd
    import pickle
    
    print("=== Iniciando ingestão de dados ===")
    
    # Configure MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    experiment_name = "relatorio-III-iris-pipeline"
    
    # Create or get experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Experimento criado com ID: {experiment_id}")
    except Exception as e:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Usando experimento existente com ID: {experiment_id}")
    
    mlflow.set_experiment(experiment_name)
    
    # Load iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')
    
    print(f"Dataset carregado: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"Classes: {iris.target_names}")
    
    # Save data for next tasks
    data_path = "/opt/airflow/dags/assets/iris_data.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump({'X': X, 'y': y, 'target_names': iris.target_names}, f)
    
    # Push metadata to XCom
    ti = context['ti']
    ti.xcom_push(key='data_path', value=data_path)
    ti.xcom_push(key='experiment_id', value=experiment_id)
    ti.xcom_push(key='experiment_name', value=experiment_name)
    ti.xcom_push(key='n_samples', value=X.shape[0])
    ti.xcom_push(key='n_features', value=X.shape[1])
    
    print("=== Ingestão de dados concluída ===")
    return data_path


def pre_processamento(**context):
    """
    Preprocessing: split data, normalize features
    """
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import mlflow
    
    print("=== Iniciando pré-processamento ===")
    
    ti = context['ti']
    data_path = ti.xcom_pull(task_ids='ingestao_dados', key='data_path')
    experiment_name = ti.xcom_pull(task_ids='ingestao_dados', key='experiment_name')
    
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X = data['X']
    y = data['y']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} amostras")
    print(f"Test set: {X_test.shape[0]} amostras")
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features normalizadas com StandardScaler")
    
    # Save preprocessed data
    preprocessed_path = "/opt/airflow/dags/assets/iris_preprocessed.pkl"
    with open(preprocessed_path, 'wb') as f:
        pickle.dump({
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'target_names': data['target_names'],
            'feature_names': X.columns.tolist()
        }, f)
    
    # Push to XCom
    ti.xcom_push(key='preprocessed_path', value=preprocessed_path)
    ti.xcom_push(key='train_size', value=len(X_train))
    ti.xcom_push(key='test_size', value=len(X_test))
    
    print("=== Pré-processamento concluído ===")
    return preprocessed_path


def treinamento(**context):
    """
    Train Random Forest model and register it in MLflow
    """
    import pickle
    import mlflow
    import mlflow.sklearn
    from sklearn.ensemble import RandomForestClassifier
    
    print("=== Iniciando treinamento ===")
    
    ti = context['ti']
    preprocessed_path = ti.xcom_pull(task_ids='pre_processamento', key='preprocessed_path')
    experiment_name = ti.xcom_pull(task_ids='ingestao_dados', key='experiment_name')
    
    # Load preprocessed data
    with open(preprocessed_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    
    # Configure MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(experiment_name)
    
    # Start MLflow run
    with mlflow.start_run(run_name="random_forest_training") as run:
        # Define model parameters
        params = {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        print("Treinando Random Forest...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        print(f"Modelo treinado com {model.n_estimators} estimadores")
        
        # Log feature importance
        feature_importance = dict(zip(data['feature_names'], model.feature_importances_))
        mlflow.log_dict(feature_importance, "feature_importance.json")
        
        # Save model locally for next tasks
        model_path = "/opt/airflow/dags/assets/random_forest_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'scaler': data['scaler'], 
                        'feature_names': data['feature_names'],
                        'target_names': data['target_names']}, f)
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        # Push to XCom
        ti.xcom_push(key='model_path', value=model_path)
        ti.xcom_push(key='run_id', value=run_id)
        ti.xcom_push(key='mlflow_tracking_uri', value="http://mlflow:5000")
    
    print("=== Treinamento concluído ===")
    return model_path


def teste(**context):
    """
    Evaluate the model: accuracy, precision, recall, f1-score
    """
    import pickle
    import mlflow
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    print("=== Iniciando teste do modelo ===")
    
    ti = context['ti']
    model_path = ti.xcom_pull(task_ids='treinamento', key='model_path')
    preprocessed_path = ti.xcom_pull(task_ids='pre_processamento', key='preprocessed_path')
    run_id = ti.xcom_pull(task_ids='treinamento', key='run_id')
    experiment_name = ti.xcom_pull(task_ids='ingestao_dados', key='experiment_name')
    
    # Load model and test data
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    with open(preprocessed_path, 'rb') as f:
        data = pickle.load(f)
    
    model = model_data['model']
    X_test = data['X_test']
    y_test = data['y_test']
    target_names = data['target_names']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Log metrics to MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log confusion matrix
        mlflow.log_dict({
            'confusion_matrix': cm.tolist(),
            'target_names': target_names.tolist()
        }, "confusion_matrix.json")
        
        # Log classification report
        mlflow.log_text(report, "classification_report.txt")
    
    # Save metrics for next tasks
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    ti.xcom_push(key='metrics', value=json.dumps(metrics))
    
    print("=== Teste do modelo concluído ===")
    return json.dumps(metrics)


def empacotamento(**context):
    """
    Convert the model to ONNX and store it in disc, create asset
    """
    import pickle
    import numpy as np
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import mlflow
    import json
    from datetime import datetime
    
    print("=== Iniciando empacotamento ===")
    
    ti = context['ti']
    model_path = ti.xcom_pull(task_ids='treinamento', key='model_path')
    run_id = ti.xcom_pull(task_ids='treinamento', key='run_id')
    experiment_name = ti.xcom_pull(task_ids='ingestao_dados', key='experiment_name')
    metrics_json = ti.xcom_pull(task_ids='teste', key='metrics')
    
    # Load model
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    feature_names = model_data['feature_names']
    n_features = len(feature_names)
    
    print(f"Convertendo modelo para ONNX ({n_features} features)...")
    
    # Define input type for ONNX
    initial_type = [('float_input', FloatTensorType([None, n_features]))]
    
    # Convert to ONNX
    try:
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12
        )
        
        # Save ONNX model to assets directory
        onnx_path = ARTIFACTS_PATH + "iris_random_forest.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"Modelo ONNX salvo em: {onnx_path}")
        
        # Log ONNX model to MLflow
        mlflow.set_tracking_uri(MLFLOW_ADDRESS)
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_id=run_id):
            mlflow.log_artifact(onnx_path, "onnx_model")
        
        # Create asset metadata file
        metrics = json.loads(metrics_json)
        asset_metadata = {
            "model_name": "iris_random_forest",
            "model_type": "RandomForestClassifier",
            "format": "ONNX",
            "created_at": datetime.now().isoformat(),
            "mlflow_run_id": run_id,
            "mlflow_experiment": experiment_name,
            "metrics": metrics,
            "features": feature_names,
            "n_features": n_features,
            "model_path": onnx_path
        }
        
        asset_path = ARTIFACTS_PATH + "model_asset_metadata.json"
        with open(asset_path, 'w') as f:
            json.dump(asset_metadata, f, indent=2)
        
        print(f"Asset metadata criado em: {asset_path}")
        
        ti.xcom_push(key='onnx_path', value=onnx_path)
        ti.xcom_push(key='asset_path', value=asset_path)
        
        print("=== Empacotamento concluído ===")
        return onnx_path
        
    except Exception as e:
        print(f"Erro ao converter para ONNX: {str(e)}")
        raise


def registro(**context):
    """
    Register the model in MLflow Model Registry
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    print("=== Iniciando registro do modelo ===")
    
    ti = context['ti']
    run_id = ti.xcom_pull(task_ids='treinamento', key='run_id')
    experiment_name = ti.xcom_pull(task_ids='ingestao_dados', key='experiment_name')
    metrics_json = ti.xcom_pull(task_ids='teste', key='metrics')
    onnx_path = ti.xcom_pull(task_ids='empacotamento', key='onnx_path')
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_ADDRESS)
    client = MlflowClient()
    
    # Register model
    model_name = "iris-random-forest-classifier"
    model_uri = f"runs:/{run_id}/model"
    
    try:
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f"Modelo registrado: {model_name}")
        print(f"Versão: {model_version.version}")
        
        # Add description
        metrics = json.loads(metrics_json)
        description = f"""
        Iris Random Forest Classifier
        
        Metrics:
        - Accuracy: {metrics['accuracy']:.4f}
        - Precision: {metrics['precision']:.4f}
        - Recall: {metrics['recall']:.4f}
        - F1-Score: {metrics['f1_score']:.4f}
        
        ONNX model available at: {onnx_path}
        """
        
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description
        )
        
        # Transition to Production if accuracy > 0.9
        if metrics['accuracy'] > 0.9:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            print(f"Modelo promovido para Production (accuracy: {metrics['accuracy']:.4f})")
        else:
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Staging"
            )
            print(f"Modelo em Staging (accuracy: {metrics['accuracy']:.4f})")
        
        ti.xcom_push(key='model_name', value=model_name)
        ti.xcom_push(key='model_version', value=model_version.version)
        
        print("=== Registro do modelo concluído ===")
        return f"{model_name}:v{model_version.version}"
        
    except Exception as e:
        print(f"Erro ao registrar modelo: {str(e)}")
        print("Continuando sem registro no Model Registry...")
        return "Registro falhou"


def build_imagem(**context):
    """
    Build Docker image for the API
    """
    print("=== Iniciando build da imagem Docker ===")
    
    docker = DockerClient(DOCKER_CONNECTION_ID).from_env()
    tag = context['params'].get('tag_imagem', 'iris-classification-api:latest')
    
    try:
        print(f"Building image with tag: {tag}")
        img, build_logs = docker.images.build(
            network_mode='host',
            path=DOCKERFILE_PATH,
            dockerfile=DOCKERFILE_PATH + "Dockerfile",
            tag=tag,
            rm=True
        )
        
        # Print build logs
        for log in build_logs:
            if 'stream' in log:
                print(log['stream'].strip())
        
        print(f"Imagem construída com sucesso: {tag}")
        
        ti = context['ti']
        ti.xcom_push(key='image_tag', value=tag)
        
        return tag
        
    except Exception as e:
        print(f"Erro ao construir imagem: {str(e)}")
        raise


def push_imagem(**context):
    """
    Push Docker image to Docker Hub
    """
    print("=== Iniciando push da imagem para Docker Hub ===")
    
    from airflow.sdk import Variable
    
    docker = DockerClient(DOCKER_CONNECTION_ID).from_env()
    ti = context['ti']
    local_tag = ti.xcom_pull(task_ids='build_imagem', key='image_tag')
    
    try:
        # Get Docker registry credentials
        username = Variable.get("docker_registry_username")
        password = Variable.get("docker_registry_password")
        
        # Ensure tag includes username for Docker Hub
        # If tag doesn't start with username/, add it
        if '/' not in local_tag.split(':')[0]:
            # Tag format: image:version -> username/image:version
            registry_tag = f"{username}/{local_tag}"
            print(f"Retagging image from '{local_tag}' to '{registry_tag}'")
            
            # Get the image and tag it for the registry
            image = docker.images.get(local_tag)
            image.tag(registry_tag)
        else:
            registry_tag = local_tag
        
        print(f"Pushing image: {registry_tag}")
        print(f"Username: {username}")
        
        # Push image
        resposta = docker.api.push(
            registry_tag,
            auth_config={'username': username, 'password': password},
            stream=True,
            decode=True
        )
        
        # Check for errors in push response
        for linha in resposta:
            if 'status' in linha:
                print(linha['status'])
            if 'errorDetail' in linha:
                raise ValueError(linha['errorDetail'])
        
        print(f"Imagem enviada com sucesso: {registry_tag}")
        
        # Store the registry tag for later use
        ti.xcom_push(key='registry_tag', value=registry_tag)
        
        return registry_tag
        
    except Exception as e:
        print(f"Erro ao enviar imagem: {str(e)}")
        raise


def verificar_container_executando(**context):
    """
    Check if container exists (running or stopped) and remove it
    """
    print("=== Verificando containers existentes ===")
    
    docker = DockerClient(DOCKER_CONNECTION_ID).from_env()
    nome = context['params'].get('nome_container', 'iris-api')
    
    try:
        # List ALL containers (running and stopped) with the same name
        containers = docker.containers.list(all=True, filters={'name': nome})
        
        if len(containers) > 0:
            for container in containers:
                print(f"Container encontrado: {container.name} (Status: {container.status})")
                print(f"Removendo container: {container.name}")
                docker.api.remove_container(container.id, force=True)
                print(f"Container {container.name} removido com sucesso")
        else:
            print(f"Nenhum container com o nome '{nome}' encontrado")
        
        return True
        
    except Exception as e:
        print(f"Erro ao verificar/remover container: {str(e)}")
        # Don't raise error if container doesn't exist
        return True


def deploy(**context):
    """
    Deploy the Docker container
    """
    print("=== Iniciando deploy do container ===")
    
    docker = DockerClient(DOCKER_CONNECTION_ID).from_env()
    ti = context['ti']
    
    # Try to get the registry tag first, then fall back to return_value, then params
    tag = (ti.xcom_pull(task_ids='push_imagem', key='registry_tag') or 
           ti.xcom_pull(task_ids='push_imagem', key='return_value') or 
           context['params'].get('tag_imagem', 'iris-classification-api:latest'))
    
    nome = context['params'].get('nome_container', 'iris-api')
    porta = context['params'].get('porta_api', 8884)
    
    # Determine the correct path for volume mount
    # When Airflow runs in container with Docker socket, we need host path
    # Check if we're running in a container by looking for /.dockerenv
    if os.path.exists('/.dockerenv'):
        # Running in container, use host path
        volume_path = ARTIFACTS_HOST_PATH
        print(f"Running in container, using host path for volume: {volume_path}")
    else:
        # Running on host, use local path
        volume_path = ARTIFACTS_PATH
        print(f"Running on host, using local path for volume: {volume_path}")
    
    # Detect Airflow's network to connect the API container to it
    network_name = None
    if os.path.exists('/.dockerenv'):
        # We're inside Airflow container, find the compose network
        hostname = os.getenv('HOSTNAME', '')
        try:
            airflow_container = docker.containers.get(hostname)
            networks = list(airflow_container.attrs['NetworkSettings']['Networks'].keys())
            # Use the first non-bridge network (usually the compose network)
            for net in networks:
                if 'default' in net.lower():
                    network_name = net
                    break
            if not network_name and networks:
                network_name = networks[0]
            print(f"Detected Airflow network: {network_name}")
        except Exception as e:
            print(f"Could not detect Airflow network: {e}")
    
    try:
        print(f"Deploying container:")
        print(f"  Image: {tag}")
        print(f"  Name: {nome}")
        print(f"  Port: {porta}:8000")
        print(f"  Assets volume: {volume_path}:/app/assets")
        if network_name:
            print(f"  Network: {network_name}")
        
        # Run container
        container = docker.containers.run(
            tag,
            name=nome,
            detach=True,
            network=network_name,  # Connect to Airflow's network
            ports={'8000/tcp': porta},
            volumes={
                volume_path: {'bind': '/app/assets', 'mode': 'ro'}
            },
            environment={
                'MODEL_METADATA_PATH': '/app/assets/model_asset_metadata.json',
                'MODEL_PATH': '/app/assets/iris_random_forest.onnx'
            }
        )
        
        print(f"Container {nome} iniciado com sucesso!")
        print(f"Container ID: {container.id}")
        print(f"API disponível em: http://localhost:{porta}")
        
        ti.xcom_push(key='container_id', value=container.id)
        ti.xcom_push(key='container_name', value=nome)
        ti.xcom_push(key='api_port', value=porta)
        
        return container.id
        
    except Exception as e:
        print(f"Erro ao fazer deploy do container: {str(e)}")
        raise


def verificar_container_saude(**context):
    """
    Verify that the container is running and API is responding
    """
    import requests
    import time
    
    print("=== Verificando saúde do container ===")
    
    docker = DockerClient(DOCKER_CONNECTION_ID).from_env()
    ti = context['ti']
    
    nome = ti.xcom_pull(task_ids='deploy', key='container_name') or context['params'].get('nome_container', 'iris-api')
    porta = ti.xcom_pull(task_ids='deploy', key='api_port') or context['params'].get('porta_api', 8884)
    
    try:
        # Check if container is running
        containers = docker.containers.list(filters={'name': nome})
        
        if len(containers) == 0:
            raise Exception(f"Container '{nome}' não está em execução!")
        
        container = containers[0]
        print(f"✓ Container encontrado: {container.name}")
        print(f"  Status: {container.status}")
        print(f"  ID: {container.id}")
        
        # Wait a bit for the API to start
        print("Aguardando API iniciar...")
        time.sleep(5)
        
        # Determine base URL
        # If running inside Airflow container (same network), use container name
        # Otherwise use localhost
        if os.path.exists('/.dockerenv'):
            # Inside Airflow container, use container name (Docker DNS)
            base_url = f"http://{nome}:8000"
            print(f"Running inside container, using container name: {base_url}")
        else:
            # On host, use localhost with mapped port
            base_url = f"http://localhost:{porta}"
            print(f"Running on host, using localhost: {base_url}")
        
        # Test root endpoint
        print(f"\nTestando endpoint raiz: {base_url}/")
        
        response = requests.get(f"{base_url}/", timeout=10)
        response.raise_for_status()
        
        root_data = response.json()
        print(f"✓ Resposta do endpoint raiz:")
        print(f"  Status: {root_data.get('status')}")
        print(f"  Model: {root_data.get('model')}")
        
        # Test metrics endpoint
        print(f"\nTestando endpoint de métricas: {base_url}/metrics")
        
        response = requests.get(f"{base_url}/metrics", timeout=10)
        response.raise_for_status()
        
        metrics_data = response.json()
        print(f"✓ Resposta do endpoint de métricas:")
        print(f"  Total predictions: {metrics_data.get('total_predictions')}")
        print(f"  Mean latency: {metrics_data.get('mean_latency_ms')} ms")
        print(f"  Model info:")
        model_info = metrics_data.get('model_info', {})
        print(f"    - Name: {model_info.get('name')}")
        print(f"    - Type: {model_info.get('type')}")
        print(f"    - Accuracy: {model_info.get('accuracy')}")
        
        # Test prediction endpoint with sample data
        print(f"\nTestando endpoint de predição: {base_url}/predict")
        
        sample_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = requests.post(f"{base_url}/predict", json=sample_data, timeout=10)
        response.raise_for_status()
        
        prediction_data = response.json()
        print(f"✓ Resposta do endpoint de predição:")
        print(f"  Prediction: {prediction_data.get('prediction')}")
        print(f"  Prediction class: {prediction_data.get('prediction_class')}")
        print(f"  Duration: {prediction_data.get('duration_ms')} ms")
        print(f"  Probabilities: {prediction_data.get('probabilities')}")
        
        print("\n=== ✓ Todos os testes passaram! API está funcionando corretamente ===")
        
        # Store test results (include both internal and external URLs)
        external_url = f"http://localhost:{porta}"
        test_results = {
            'container_running': True,
            'container_status': container.status,
            'root_endpoint': 'OK',
            'metrics_endpoint': 'OK',
            'predict_endpoint': 'OK',
            'sample_prediction': prediction_data.get('prediction'),
            'api_url_internal': base_url,
            'api_url_external': external_url
        }
        
        ti.xcom_push(key='health_check_results', value=json.dumps(test_results))
        
        return test_results
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Erro ao testar API: {str(e)}")
        raise Exception(f"API não está respondendo corretamente: {str(e)}")
    except Exception as e:
        print(f"✗ Erro ao verificar container: {str(e)}")
        raise


# Define DAG
dag = DAG(
    "relatorio-III",
    schedule=None,
    description="Pipeline completo de ML com Iris dataset usando MLflow e ONNX",
    default_args={
        'depends_on_past': False,
        'owner': 'airflow'
    },
    params={
        'tag_imagem': 'iris-classification-api:latest',
        'nome_container': 'iris-api',
        'porta_api': 8884
    },
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['mlflow', 'iris', 'random-forest', 'onnx', 'docker', 'api']
)

with dag:
    # ML Pipeline tasks
    task_ingestao = PythonOperator(
        task_id="ingestao_dados",
        python_callable=ingestao_dados
    )
    
    task_preprocessamento = PythonOperator(
        task_id="pre_processamento",
        python_callable=pre_processamento
    )
    
    task_treinamento = PythonOperator(
        task_id="treinamento",
        python_callable=treinamento
    )
    
    task_teste = PythonOperator(
        task_id="teste",
        python_callable=teste
    )
    
    task_empacotamento = PythonOperator(
        task_id="empacotamento",
        python_callable=empacotamento
    )
    
    task_registro = PythonOperator(
        task_id="registro",
        python_callable=registro
    )
    
    # Docker deployment tasks
    task_build_imagem = PythonOperator(
        task_id="build_imagem",
        python_callable=build_imagem
    )
    
    task_push_imagem = PythonOperator(
        task_id="push_imagem",
        python_callable=push_imagem
    )
    
    task_verificar_container = PythonOperator(
        task_id="verificar_container_executando",
        python_callable=verificar_container_executando
    )
    
    task_deploy = PythonOperator(
        task_id="deploy",
        python_callable=deploy
    )
    
    task_verificar_saude = PythonOperator(
        task_id="verificar_container_saude",
        python_callable=verificar_container_saude
    )

    # Define dependencies
    # ML Pipeline: ingestao -> preprocessamento -> treinamento -> teste -> empacotamento -> registro
    task_ingestao >> task_preprocessamento >> task_treinamento >> task_teste >> task_empacotamento >> task_registro
    
    # Docker Deployment: registro -> build -> push -> verificar -> deploy -> verificar_saude
    task_registro >> task_build_imagem >> task_push_imagem >> task_verificar_container >> task_deploy >> task_verificar_saude 

from airflow.sdk import dag, task, task_group
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import requests
from airflow.hooks.base import BaseHook
from docker import DockerClient as DockerClientAPI

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
DOCKERFILE_PATH = "/opt/airflow/dags/ci-cd-completo/api/"
ARTIFACTS_PATH = "/opt/airflow/dags/ci-cd-completo/api/artifacts/"
MLFLOW_ADDRESS = "http://mlflow:5000/"

@dag("dag_ci_cd", schedule=None, params={"nome_modelo": "aula-16-10-25", "tag_imagem": "lucianei/mlops-16-10"})
def dag_ci_cd():
    @task
    def criar_experimento(**context):
        timestamp = context['ts']

        if 'nome_modelo' not in context['params']:
            raise KeyError("Está faltando o parâmetro:", 'nome_modelo')

        # Cria cliente para comunicar com o MLFlow
        import requests
        
        response = requests.post(
            f"{MLFLOW_ADDRESS}api/2.0/mlflow/experiments/create",
            json={'name': timestamp + '_dag_ci_cd'}
        )
        informacao_experimento = response.json()
        return informacao_experimento['experiment_id']
    
    @task.branch(task_id='verificar_experimento')
    def verificar_experimento_criado(resultado: dict):
        if 'experiment_id' in resultado:
            return ['configurar_mlflow']
        return ['erro']

    @task
    def consumir_dataset():
        import pandas as pd

        df = pd.read_csv("/opt/airflow/dags/datasets/students.csv")

        df = df.drop([
            'Timestamp', 
            "What coping strategy you use as a student?",
            "Do you have any bad habits like smoking, drinking on a daily basis?", 
            "What would you rate the academic  competition in your student life"], 
        axis=1)

        return df
    
    def hot_encode_academic_stage(linha):
        if linha == 'high school':
            return 0
        elif linha == 'undergraduate':
            return 1
        elif linha == 'post-graduate':
            return 2

    def hot_encode_study_environment(linha):
        if linha == 'Peaceful':
            return 0
        elif linha == 'Noisy':
            return 1
        elif linha == 'disrupted':
            return 2

    @task
    def transformar_dados(experiment_id, df):
        import mlflow
        import pickle
        mlflow.set_tracking_uri(MLFLOW_ADDRESS)

        from sklearn.preprocessing import MinMaxScaler
        from sklearn.model_selection import train_test_split

        mlflow.sklearn.autolog()

        with mlflow.start_run(experiment_id=experiment_id, run_name="scale") as run:
            # 1. Hot encoding das features
            df['Your Academic Stage'] = df['Your Academic Stage'].apply(hot_encode_academic_stage) 
            df['Study Environment'] = df['Study Environment'].apply(hot_encode_study_environment)

            colunas = [
                'Your Academic Stage', 
                'Peer pressure', 
                'Academic pressure from your home', 
                'Study Environment', 
                'Rate your academic stress index'
            ]

            # 2. Separar em treino e teste
            df_treino, df_teste = train_test_split(df, test_size=0.3, random_state=41)

            alvo = 'Rate your academic stress index'

            # 3. Normalizar dados
            scaler = MinMaxScaler()

            features = colunas[:-1]
            
            df_treino[features] = scaler.fit_transform(df_treino[features])
            df_teste[features] = scaler.transform(df_teste[features])

            mlflow.sklearn.log_model(scaler, artifact_path="scaler")
            mlflow.log_metrics({'n_samples_seen_': scaler.n_samples_seen_})

            # 4. Serializa e salva scaler
            with open(ARTIFACTS_PATH + "scaler.pkl", 'wb') as arquivo:
                arquivo.write(pickle.dumps(scaler))

            # 5. Retornar treino e teste
            return {'treino': df_treino, 'teste': df_teste, 'alvo': alvo}
    
    def serializar_modelo(modelo, shape_x):
        tipo_inicial = [('float_type', FloatTensorType(shape_x))]

        return convert_sklearn(modelo, initial_types=tipo_inicial).SerializeToString()

    @task
    def treinamento(experiment_id, dados_transformados, **context):
        nome_modelo = context['params']['nome_modelo']
        
        # Debug: Print what we received
        print(f"DEBUG: dados_transformados keys: {dados_transformados.keys()}")
        print(f"DEBUG: dados_transformados type: {type(dados_transformados)}")

        import mlflow
        mlflow.set_tracking_uri(MLFLOW_ADDRESS)

        from sklearn.tree import DecisionTreeClassifier

        mlflow.sklearn.autolog()

        with mlflow.start_run(experiment_id=experiment_id, run_name="treinamento_modelo") as run:
            df_treino = dados_transformados['treino']
            alvo = dados_transformados['alvo']
            
            # Debug: Print DataFrame info
            print(f"DEBUG: df_treino shape: {df_treino.shape}")
            print(f"DEBUG: df_treino columns: {df_treino.columns.tolist()}")
            print(f"DEBUG: alvo: {alvo}")

            X_train, y_train = df_treino.drop(alvo, axis=1), df_treino[alvo]

            dt = DecisionTreeClassifier()

            dt.fit(X_train, y_train)

            # Registra modelo no MLFlow
            mlflow.sklearn.log_model(dt, input_example=X_train, name=nome_modelo, registered_model_name=nome_modelo)

            # Retorna formato do dataset
            return X_train.shape[1]

    @task
    def teste(experiment_id, dados_transformados, **context):
        nome_modelo = context['params']['nome_modelo']
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_ADDRESS)

        mlflow.sklearn.autolog()

        with mlflow.start_run(experiment_id=experiment_id, run_name="teste_modelo") as run:
            modelo = mlflow.sklearn.load_model(f"models:/{nome_modelo}/latest")
            df_teste = dados_transformados['teste']
            alvo = dados_transformados['alvo']

            X_test, y_test = df_teste.drop(alvo, axis=1), df_teste[alvo]

            return modelo.score(X_test, y_test)

    @task
    def empacotamento(experiment_id, formato_entrada_modelo, **context):
        nome_modelo = context['params']['nome_modelo']
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_ADDRESS)

        from mlflow import start_run
        from mlflow.sklearn import load_model
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        with open(ARTIFACTS_PATH + f"{nome_modelo}.onnx", 'wb') as arquivo, start_run(experiment_id=experiment_id, run_name="empacotamento") as run:
            modelo = load_model(f"models:/{nome_modelo}/latest")

            tipos = [('float', FloatTensorType([1, formato_entrada_modelo]))]

            modelo_serializado = convert_sklearn(modelo, initial_types=tipos)

            arquivo.write(modelo_serializado.SerializeToString())

    @task_group("build")
    def build():
        experiment_id = criar_experimento()
        dataframe = consumir_dataset()
        
        resultado = transformar_dados(experiment_id, dataframe)
        formato_entrada_modelo = treinamento(experiment_id, resultado)

        experiment_id >> \
            dataframe >> \
            resultado >> \
            formato_entrada_modelo >> \
            teste(experiment_id, resultado) >> \
            empacotamento(experiment_id, formato_entrada_modelo)

    @task_group("entrega")
    def entrega():
        print('Iniciando entrega da aplicação com Docker', DOCKER_CONNECTION_ID)
        docker = DockerClient(DOCKER_CONNECTION_ID).from_env()

        @task 
        def build_imagem(**context):
            tag = context['params']['tag_imagem']
            img = docker.images.build(network_mode='host', path=DOCKERFILE_PATH, dockerfile=DOCKERFILE_PATH + "Dockerfile", tag=tag)

            return tag

        @task
        def push_imagem(tag):
            from airflow.sdk import Variable

            username = Variable.get("docker_registry_username")
            password = Variable.get("docker_registry_password")
            print('DEBUG: TAG', tag)
            print('DEBUG: USERNAME', username)
            resposta = docker.api.push(tag, auth_config={'username': username, 'password': password}, stream=True, decode=True)

            for linha in resposta:
                if 'errorDetail' in linha.keys():
                    raise ValueError(linha['errorDetail']) 

            return tag

        tag = build_imagem()
        push_imagem(tag)

    @task 
    def verificar_container_executando(**context):
        docker = DockerClient(DOCKER_CONNECTION_ID).from_env()
        nome = context['params']['nome_modelo']

        containers = docker.containers.list(filters={'name': nome})

        if len(containers) == 1:
            print("Apagando container", containers[0])

            return docker.api.remove_container(nome, force=True)

    @task
    def deploy(**context):
        tag = context['params']['tag_imagem']
        nome = context['params']['nome_modelo']

        docker = DockerClient(DOCKER_CONNECTION_ID).from_env()

        docker.containers.run(tag, name=nome, stdout=False, detach=True, ports={'80/tcp': [8883]})

    @task
    def erro(mensagem: dict):
        print(mensagem)
    
    build() >> entrega() >> verificar_container_executando() >> deploy()
    
dag_ci_cd()
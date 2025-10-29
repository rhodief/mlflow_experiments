from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime
import json


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
    experiment_name = "relatorio-II-iris-pipeline"
    
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
        
        # Save ONNX model
        onnx_path = "/opt/airflow/dags/assets/iris_random_forest.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"Modelo ONNX salvo em: {onnx_path}")
        
        # Log ONNX model to MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
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
        
        asset_path = "/opt/airflow/dags/assets/model_asset_metadata.json"
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
    mlflow.set_tracking_uri("http://mlflow:5000")
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


# Define DAG
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
    tags=['mlflow', 'iris', 'random-forest', 'onnx']
)

with dag:
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

    # Define dependencies
    task_ingestao >> task_preprocessamento >> task_treinamento >> task_teste >> task_empacotamento >> task_registro 

# Entrega 02

- Relatório PDF 
    - nome dos integrantes
    - nome dag: relatorio-II
    - 6 etapas
        - ingestão de dados: IRIS
        - Dados Treino/Teste: pre-processamento
        - Treinamento: Treinamento com modelo (random forrest) e registro MLFlow
        - Teste: Avaliar e registrar MLFLow... (acurácia, precisão, revocação, f1-score)
        - Empacotamento: converter para ONNX
        - Regis4tro: Registrar MLFlow

    - print da DAG com etapas funcionando
    - print do experimento MLFlow com runs do experimento. 

# Prompt =D

Using Airflow 3 and MLFlow3 create a experiment and a DAG named "relatorio-II" using PythonOperator with the following steps:

# "ingestao_dados"
it should load the "iris" dataset from scikit learning and setup a new experiment in MLFlow accoudinally. 

# "pre-processamento"
preprocessing stuff with considering normalize if it fits

   
# "treinamento"
Use Random forrest and regist the model in MLFlow Acoundinally

# "teste": 
Evaluate the accuracy of the model and set it up in MLFLOW: accuracy, precison, recall, f1-score

# "empacotamento": 
Convert the model to ONNX and store it in disc. At the end, it should create a new asset in folder "assets"

# "registro":
Make the registry in MLFLow


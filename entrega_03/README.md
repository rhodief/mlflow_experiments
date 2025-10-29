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

Create in api folder a main.py with a FastAPI app and load the model in assets folder in omnx format (see dag_entrega_3.py to know how it was trained).
The app should have this routes:

/predict (método POST): receive in the body a json with the feature properties and returns a json with the predicion and the duration of the prediction;

/metrics (método GET): Retruns the total amount of predictions and the mean latancy by predictions. 

Setup also the swagger documentation accourdinally

In order to this app to work, create in Dockerfile (api folder) for the api, install the dependecies for the api and the model. create sh to build the image and another to launch the app (docker run)
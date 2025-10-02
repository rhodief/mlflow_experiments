from airflow.sdk import asset, dag, task

import requests

@asset(uri="file:///opt/airflow/dags/students.csv", name="students.csv", schedule="@daily")
def baixar_dataset():
    requisicao = requests.get("https://prhrck.com/files/students.csv")
    dados = requisicao.text

    with open("/opt/airflow/dags/students.csv", "w") as arquivo:
        arquivo.write(dados)

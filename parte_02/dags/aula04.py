from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator

def ingestao_dados():
    print("Executando tarefa!")
    caminho_dataset = "/opt/airflow/dags"
    return caminho_dataset

def treinamento(ti=None):
    print("Treinando modelo")

    if ti == None:
        return "Não consegui acessar o contexto da tarefa!"

    valor = ti.xcom_pull(task_ids="ingestao", key="return_value")

    return "OK " + valor

def teste():
    print("Testando modelo")
    return "OK"

def empacotamento():
    print("Empacotando modelo")
    return "OK"

def registro():
    print("Registrando modelo")
    return "OK"

dag = DAG("aula04", schedule=None, description="DAG aula04", default_args={'depends_on_past': False})

with dag:
    ingestao = PythonOperator(task_id="ingestao", python_callable=ingestao_dados)
    op_treinamento = PythonOperator(task_id="treinamento", python_callable=treinamento)
    op_teste = PythonOperator(task_id="teste", python_callable=teste)
    op_empacotamento = PythonOperator(task_id="empacotamento", python_callable=empacotamento)
    op_registro = PythonOperator(task_id="registro", python_callable=registro)

    ingestao >> op_treinamento >> op_teste >> op_empacotamento >> op_registro 

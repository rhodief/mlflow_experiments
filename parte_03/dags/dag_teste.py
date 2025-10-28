from airflow.sdk import dag, task
from time import sleep

@dag("dag_simples", catchup=False)
def dag_simples():
    
    @task()
    def tarefa1():
        sleep(5)
        return "OK Tarefa1"

    @task()
    def tarefa2(msg):
        print("Recebido da tarefa anterior:", msg)
        sleep(3)
        return "OK Tarefa2"
    
    resultado_t1 = tarefa1()
    resultado_t2 = tarefa2(resultado_t1)

dag_simples()

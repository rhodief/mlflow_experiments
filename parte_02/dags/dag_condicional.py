from airflow.sdk import dag, task

@dag("dag_condicional", catchup=False)
def dag_condicional():
    
    @task()
    def tarefa1():
        return "tarefa1"

    @task.branch(task_id="condicao")
    def condicao(resultado):
        if resultado == "tarefa1":
            return ["ok"]
        return ["erro"]
    
    @task()
    def ok():
        print("Deu tudo certo!")
    
    @task()
    def erro():
        print("ERRO!")

    ret1 = tarefa1()

    condicao(ret1) >> [ok(), erro()]

dag_condicional()

from airflow.sdk import dag, task

@dag("aula04_taskflow", catchup=False)
def aula04_taskflow():

    @task()
    def ingestao_dados():
        print("Executando tarefa!")
        caminho_dataset = "/opt/airflow/dags"
        return caminho_dataset

    @task()
    def treinamento(contexto_ingestao):
        print("Treinando modelo")

        if contexto_ingestao == None:
            return "Não consegui acessar o contexto da tarefa!"

        return "OK " + contexto_ingestao

    @task()
    def teste():
        print("Testando modelo")
        return "OK"
    
    @task()
    def empacotamento():
        print("Empacotando modelo")
        return "OK"
    
    @task()
    def registro():
        print("Registrando modelo")
        return "OK"
    
    resultado_ingestao = ingestao_dados()
    resultado_treinamento = treinamento(resultado_ingestao)

    resultado_treinamento >> teste() >> empacotamento() >> registro()

aula04_taskflow()
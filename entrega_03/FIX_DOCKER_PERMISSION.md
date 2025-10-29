# Fix: Docker Permission Denied Error - RESOLVED ✅

## Problema

O erro `PermissionError(13, 'Permission denied')` ocorre quando o Airflow tenta acessar o Docker socket mas não tem permissões adequadas.

```
DockerException: Error while fetching server API version: ('Connection aborted.', PermissionError(13, 'Permission denied'))
```

## Causa

O container Airflow precisa ter acesso ao socket Docker (`/var/run/docker.sock`) para construir e executar containers. Por padrão, apenas usuários do grupo `docker` têm essa permissão.

## Solução Implementada ✅

A solução utiliza `group_add` no docker-compose para adicionar o grupo Docker ao usuário do Airflow sem alterar o GID principal (que deve permanecer como 0 para o Airflow funcionar corretamente).

### Alterações Feitas

#### 1. init.cluster.sh

```bash
export AIRFLOW_UID=$(id -u)
export AIRFLOW_GID=$(id -g)
export DOCKER_GID=$(getent group docker | cut -d: -f3)  # ← Captura o GID do Docker
docker compose up -d
```

#### 2. docker-compose.yml

Adicionado `group_add` para incluir o grupo Docker:

```yaml
x-airflow-common:
  &airflow-common
  image: ${AIRFLOW_IMAGE_NAME:-apache/airflow:3.0.6}
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
  user: "${AIRFLOW_UID:-50000}:0"  # GID deve ser 0 para Airflow
  group_add:
    - "${DOCKER_GID:-999}"  # ← Adiciona grupo Docker aos grupos suplementares
```

**Por que `group_add` em vez de mudar o GID?**

O Airflow requer GID=0 para funcionar corretamente. Usar `group_add` adiciona o grupo Docker como grupo suplementar, permitindo acesso ao socket Docker sem quebrar o Airflow.

## Como Aplicar a Correção

### 1. Parar o Cluster

```bash
cd /home/rhodie/dev/lab/mlflow/entrega_03
docker compose down
```

### 2. Reiniciar com o Script Atualizado

```bash
./init.cluster.sh
```

### 3. Verificar as Permissões ✅

```bash
# Verificar grupos do usuário Airflow no container
docker exec entrega_03-airflow-worker-1 groups
# Saída esperada: root 998 (ou o GID do seu grupo Docker)

# Testar acesso ao Docker
docker exec entrega_03-airflow-worker-1 docker ps
# Deve listar os containers sem erro!
```

## Teste Completo ✅

Após aplicar a correção, execute este teste:

1. Acesse o Airflow UI: http://localhost:8080
   - User: `airflow`
   - Password: `airflow`

2. Encontre o DAG `relatorio-II`

3. Configure os parâmetros (opcional):
   - `tag_imagem`: Sua tag Docker (ex: `seunome/iris-api:latest`)
   - `nome_container`: Nome do container (ex: `iris-api`)
   - `porta_api`: Porta da API (ex: `8884`)

4. Trigger o DAG

5. A task `build_imagem` agora deve funcionar!

## Variáveis do Airflow

Não esqueça de configurar as credenciais do Docker Hub:

### Via UI

1. Acesse: Admin → Variables
2. Adicione:
   - **Key**: `docker_registry_username` → **Value**: seu usuário
   - **Key**: `docker_registry_password` → **Value**: sua senha

### Via CLI

```bash
docker exec entrega_03-airflow-worker-1 airflow variables set docker_registry_username "seu_usuario"
docker exec entrega_03-airflow-worker-1 airflow variables set docker_registry_password "sua_senha"
```

### Via Arquivo .env

Crie um arquivo `.env` no diretório `entrega_03`:

```bash
DOCKER_REGISTRY_USERNAME=seu_usuario
DOCKER_REGISTRY_PASSWORD=sua_senha
```

## Troubleshooting

### Ainda recebo erro de permissão

1. Verifique se você está no grupo Docker:
   ```bash
   groups
   ```

2. Se não estiver, adicione-se:
   ```bash
   sudo usermod -aG docker $USER
   newgrp docker  # ou faça logout/login
   ```

3. Reinicie o cluster:
   ```bash
   ./init.cluster.sh
   ```

### Docker socket não existe

```bash
ls -la /var/run/docker.sock
```

Se não existir, certifique-se de que o Docker está rodando:
```bash
sudo systemctl start docker
```

### Erro "Cannot connect to Docker daemon"

Verifique se o Docker está rodando:
```bash
docker ps
```

Se não estiver:
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

## Segurança

⚠️ **Nota de Segurança**: Dar acesso ao Docker socket é equivalente a dar acesso root ao host. Isso é aceitável para desenvolvimento local, mas **NÃO deve ser usado em produção** sem medidas de segurança adicionais.

Para produção, considere:
- Docker-in-Docker (DinD)
- Kubernetes com RBAC
- Serviços gerenciados de CI/CD
- Soluções como Kaniko para builds sem privilégios

## Verificação Final

Depois de aplicar as correções e reiniciar:

```bash
# 1. Verificar se containers estão rodando
docker ps

# 2. Verificar logs do worker
docker logs entrega_03-airflow-worker-1 | grep -i docker

# 3. Testar acesso ao Docker
docker exec entrega_03-airflow-worker-1 docker version

# 4. Executar o DAG e verificar logs
# (via Airflow UI)
```

Se todos os passos acima funcionarem, o problema está resolvido! 🎉

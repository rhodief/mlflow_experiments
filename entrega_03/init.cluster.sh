export AIRFLOW_UID=$(id -u)
export AIRFLOW_GID=$(id -g)
export DOCKER_GID=$(getent group docker | cut -d: -f3)
docker compose up -d

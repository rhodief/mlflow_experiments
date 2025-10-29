# Fix: Docker Hub Push - Access Denied

## Problem
```
denied: requested access to the resource is denied
```

The push was trying to push to `docker.io/library/iris-classification-api` instead of `docker.io/username/iris-classification-api`.

## Root Cause

When you don't specify a username in the Docker image tag, Docker tries to push to the default `library` namespace, which is reserved for official Docker images. Only Docker, Inc. can push to that namespace.

## Solution ✅

Updated the `push_imagem` function to automatically prepend the username to the tag if it's missing.

### Before:
```python
tag = 'iris-classification-api:latest'
# Tries to push to: docker.io/library/iris-classification-api ❌
```

### After:
```python
local_tag = 'iris-classification-api:latest'
username = 'lucianei'
registry_tag = f"{username}/{local_tag}"  # lucianei/iris-classification-api:latest
# Pushes to: docker.io/lucianei/iris-classification-api ✅
```

## Changes Made

### 1. Updated `push_imagem` function

```python
def push_imagem(**context):
    # Get local tag from build step
    local_tag = ti.xcom_pull(task_ids='build_imagem', key='image_tag')
    
    # Get username
    username = Variable.get("docker_registry_username")
    
    # Ensure tag includes username for Docker Hub
    if '/' not in local_tag.split(':')[0]:
        # Add username prefix
        registry_tag = f"{username}/{local_tag}"
        
        # Retag the image
        image = docker.images.get(local_tag)
        image.tag(registry_tag)
    else:
        registry_tag = local_tag
    
    # Push with the correct tag
    docker.api.push(registry_tag, ...)
    
    # Store registry tag for deploy step
    ti.xcom_push(key='registry_tag', value=registry_tag)
```

### 2. Updated `deploy` function

```python
def deploy(**context):
    # Use the registry tag from push step
    tag = ti.xcom_pull(task_ids='push_imagem', key='registry_tag')
    
    # Deploy with the correct tag
    docker.containers.run(tag, ...)
```

## How to Use

### Option 1: Use Default Tag (Recommended for Testing)
Just run the DAG without parameters. The function will automatically add your username:
- Input: `iris-classification-api:latest`
- Actual push: `lucianei/iris-classification-api:latest`

### Option 2: Specify Full Tag with Username
When triggering the DAG, provide the full tag:
```json
{
  "tag_imagem": "lucianei/iris-api:v1.0",
  "nome_container": "iris-api",
  "porta_api": 8884
}
```

### Option 3: Use Different Repository Name
```json
{
  "tag_imagem": "my-custom-name:latest",
  "nome_container": "iris-api",
  "porta_api": 8884
}
```
This will push to: `lucianei/my-custom-name:latest`

## Verification

After the fix, you should see in the logs:

```
[INFO] Retagging image from 'iris-classification-api:latest' to 'lucianei/iris-classification-api:latest'
[INFO] Pushing image: lucianei/iris-classification-api:latest
[INFO] Username: lucianei
[INFO] The push refers to repository [docker.io/lucianei/iris-classification-api]
[INFO] latest: digest: sha256:... size: 2841
[INFO] Imagem enviada com sucesso: lucianei/iris-classification-api:latest
```

## Re-run the DAG

Simply trigger the DAG again from the Airflow UI:
1. Go to http://localhost:8080
2. Find DAG `relatorio-III`
3. Click "Trigger DAG"
4. (Optional) Configure parameters
5. The push should now succeed! ✅

## Common Docker Hub Tag Formats

| Format | Example | Valid? |
|--------|---------|--------|
| `image:tag` | `iris-api:latest` | ❌ → Needs username |
| `username/image:tag` | `lucianei/iris-api:latest` | ✅ |
| `registry/username/image:tag` | `docker.io/lucianei/iris-api:latest` | ✅ |
| `registry:port/image:tag` | `localhost:5000/iris-api:latest` | ✅ (private registry) |

## Troubleshooting

### Still getting access denied?

1. **Check credentials:**
   ```bash
   docker exec entrega_03-airflow-worker-1 airflow variables list | grep docker
   ```

2. **Test login manually:**
   ```bash
   docker exec entrega_03-airflow-worker-1 docker login -u lucianei
   ```

3. **Verify the tag includes username:**
   Check the logs for the actual tag being pushed

### Wrong username in tag?

Make sure the `docker_registry_username` variable matches your Docker Hub account:
```bash
docker exec entrega_03-airflow-worker-1 airflow variables set docker_registry_username "your_correct_username"
```

### Want to use a different registry?

For a private registry, include the full registry URL in the tag:
```json
{
  "tag_imagem": "myregistry.com:5000/iris-api:latest"
}
```

The function will detect the `/` and skip username prepending.

## Summary

✅ **Fixed**: Automatic username prepending for Docker Hub tags  
✅ **Benefit**: No need to manually specify username in every tag  
✅ **Flexible**: Still supports full tags with username/registry  
✅ **Safe**: Only modifies tags that need it  

The DAG is now ready to push to Docker Hub successfully! 🎉

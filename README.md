# Ollama FastAPI Server (v2.0 - Fly.io Ready)

Self-hosted LLM API with API key authentication.

## Deploy to Fly.io

### 1. Install flyctl and login
```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

### 2. Launch app
```bash
fly launch --name ollama-fastapi-railway-deployment --region iad --no-deploy
```

### 3. Set secrets
```bash
fly secrets set MASTER_KEY=your-strong-master-key-here
```

### 4. Create Persistence (IMPORTANT)
To avoid losing models on restart, you **must** create the storage volumes before deploying:
```bash
fly volume create ollama_data --region iad --count 2 --size 10
```

### 5. Deploy
```bash
fly deploy
```

## Fly.io Free Tier Limits
- **Memory**: 2GB max (this config is optimized for it)
- **CPU**: 1 shared core (optimized for stability)
- **Model**: Use qwen2.5:0.5b (~300MB) or tinyllama (~600MB)
- **Storage**: Persistent via Volumes (models stay saved across restarts)

## Authentication

### Create API Key (needs MASTER_KEY)
```bash
curl -X POST https://your-app.fly.dev/admin/keys \
  -H "X-Master-Key: your-master-key" \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app"}'
```

### Use API Key
```python
import requests

url = "https://your-app.fly.dev/v1/chat/completions"
headers = {
    "Authorization": "Bearer ollama_xxxxxxxx",
    "Content-Type": "application/json"
}
data = {
    "model": "qwen2.5:0.5b",
    "messages": [{"role": "user", "content": "Hello!"}]
}
res = requests.post(url, json=data, headers=headers)
print(res.json())
```

## Pages
- /ui - Dashboard
- /api-docs - API reference
- /docs - Swagger UI
- /redoc - ReDoc
- /health - Status (no auth)

## Models That Fit in 2GB
| Model | Size | Works? |
|-------|------|--------|
| qwen2.5:0.5b | ~300MB | Yes |
| tinyllama | ~600MB | Yes |
| phi3:mini | ~2GB | Maybe (tight) |
| llama3.2:1b | ~1.3GB | Maybe |

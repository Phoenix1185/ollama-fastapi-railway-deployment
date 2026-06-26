import os
import time
import secrets
import hashlib
import httpx
import asyncio
import json
import logging
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, String, Integer, BigInteger, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ollama API Server",
    description="Self-hosted LLM API with Ollama + FastAPI. API key protected.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "tinyllama:latest")
MASTER_KEY = os.getenv("MASTER_KEY", "ollama-master-key-change-me")
DATABASE_URL = os.getenv("DATABASE_URL")

# Azure OpenAI Configuration
AZURE_OPENAI_MODEL_ENV = os.getenv("AZURE_OPENAI_MODEL", "").strip()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_KEY = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY") or "").strip()
AZURE_OPENAI_DEPLOYMENTS = os.getenv("AZURE_OPENAI_DEPLOYMENTS", "gpt-4o-mini").strip().split(",")
AZURE_OPENAI_DEPLOYMENTS = [d.strip() for d in AZURE_OPENAI_DEPLOYMENTS if d.strip()]
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()

if AZURE_OPENAI_MODEL_ENV.startswith("http"):
    if "/openai/v1" in AZURE_OPENAI_MODEL_ENV:
        AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_MODEL_ENV.split("/openai/v1")[0]
    elif "/openai" in AZURE_OPENAI_MODEL_ENV:
        AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_MODEL_ENV.split("/openai")[0]

USE_AZURE = bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY)

if USE_AZURE:
    from openai import AzureOpenAI
    try:
        azure_client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        logger.info(f"Azure OpenAI client initialized with endpoint: {AZURE_OPENAI_ENDPOINT}")
    except Exception as e:
        logger.error(f"Failed to initialize Azure client: {e}")
        azure_client = None
        USE_AZURE = False
else:
    azure_client = None

# Database Setup
Base = declarative_base()

class APIKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True)
    key_hash = Column(String(64), unique=True, index=True)
    name = Column(String(100))
    created_at = Column(BigInteger)

engine = None
SessionLocal = None

def init_db():
    global engine, SessionLocal
    if not DATABASE_URL:
        logger.error("DATABASE_URL not set. API key management will fail.")
        return
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(keep_warm_background())

async def keep_warm_background():
    """Aggressive background task that pings Ollama every 2 minutes with actual inference."""
    await asyncio.sleep(30)  # Start sooner
    while True:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Use actual chat endpoint for the heartbeat to ensure model stays in GPU/RAM
                payload = {
                    "model": DEFAULT_MODEL,
                    "messages": [{"role": "user", "content": "heartbeat"}],
                    "stream": False,
                    "options": {"num_predict": 1}
                }
                await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
                logger.info(f"Aggressive keep-warm heartbeat sent for {DEFAULT_MODEL}")
        except Exception as e:
            logger.warning(f"Keep-warm heartbeat failed: {e}")
        await asyncio.sleep(120)  # Ping every 2 minutes (more frequent)

def get_db():
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Security
security = HTTPBearer(auto_error=False)

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security), db = Depends(get_db)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header. Use: Bearer YOUR_API_KEY")
    token = credentials.credentials
    
    # Allow Master Key for all endpoints
    if token == MASTER_KEY:
        return token
        
    key_hash = hash_key(token)
    key_record = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
    if not key_record:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token

def verify_master_key(x_master_key: str = Header(None)):
    if not x_master_key:
        raise HTTPException(status_code=401, detail="Missing X-Master-Key header")
    if x_master_key != MASTER_KEY:
        raise HTTPException(status_code=403, detail="Invalid master key")
    return x_master_key

# Request Models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class GenerateRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class PullModelRequest(BaseModel):
    name: str

class CreateKeyRequest(BaseModel):
    name: str

class RevokeKeyRequest(BaseModel):
    key_hash: str

# Endpoints
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code == 200:
                return {"status": "ok", "ollama": "connected", "auth": "enabled"}
    except:
        pass
    return {"status": "degraded", "ollama": "not ready", "auth": "enabled"}

@app.get("/ping")
async def ping():
    """Lightweight keep-alive endpoint for external pingers. No auth required."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            if r.status_code == 200:
                return {"status": "alive", "timestamp": int(time.time())}
    except:
        pass
    return {"status": "starting", "timestamp": int(time.time())}

@app.post("/warmup")
async def warmup():
    """Send a tiny inference to keep the model loaded in RAM. No auth required."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "options": {"num_predict": 1}
            }
            r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            return {"status": "warm", "model": DEFAULT_MODEL, "timestamp": int(time.time())}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
@app.get("/UI", response_class=HTMLResponse)
async def web_ui():
    return HTML_CONTENT

@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs():
    return API_DOCS_CONTENT

@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    models = []
    
    if USE_AZURE:
        for deployment in AZURE_OPENAI_DEPLOYMENTS:
            models.append({
                "id": deployment,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "azure"
            })
        
    # Add Ollama models
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code == 200:
                data = r.json()
                for m in data.get("models", []):
                    models.append({
                        "id": m["name"],
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "ollama"
                    })
    except Exception as e:
        logger.error(f"Ollama tags error: {e}")
        if not models:
            return {"object": "list", "data": []}
            
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    requested_model = req.model or DEFAULT_MODEL
    use_azure_for_this = USE_AZURE and requested_model in AZURE_OPENAI_DEPLOYMENTS
    
    if use_azure_for_this:
        if not azure_client:
            raise HTTPException(status_code=500, detail="Azure client not initialized.")
        try:
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            logger.info(f"Sending request to Azure OpenAI: {requested_model}")
            
            if req.stream:
                response = azure_client.chat.completions.create(
                    model=requested_model,
                    messages=messages,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    stream=True
                )
                
                async def azure_streamer():
                    try:
                        for chunk in response:
                            if chunk.choices:
                                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logger.error(f"Azure streaming error: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return StreamingResponse(azure_streamer(), media_type="text/event-stream")
            else:
                response = azure_client.chat.completions.create(
                    model=requested_model,
                    messages=messages,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    stream=False
                )
                return response.model_dump()
        except Exception as e:
            logger.error(f"Azure OpenAI error: {e}")
            raise HTTPException(status_code=500, detail=f"Azure error: {str(e)}")
            
    # Fallback to Ollama
    model = requested_model
    try:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": req.stream,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens
            }
        }
        
        if req.stream:
            async def streamer():
                async with httpx.AsyncClient(timeout=None) as client:
                    try:
                        async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as response:
                            async for line in response.aiter_lines():
                                if line:
                                    data = json.loads(line)
                                    chunk = {
                                        "id": f"chatcmpl-{int(time.time())}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": data.get("message", {}).get("content", "")},
                                            "finish_reason": "stop" if data.get("done") else None
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return StreamingResponse(streamer(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
                data = r.json()
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": data.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }]
                }
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/api/generate")
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    model = req.model or DEFAULT_MODEL
    try:
        payload = {
            "model": model,
            "prompt": req.prompt,
            "stream": req.stream,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens
            }
        }
        
        if req.stream:
            async def streamer():
                async with httpx.AsyncClient(timeout=None) as client:
                    try:
                        async with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=payload) as response:
                            async for line in response.aiter_lines():
                                if line:
                                    yield f"{line}\n"
                    except Exception as e:
                        yield json.dumps({"error": str(e)})
            return StreamingResponse(streamer(), media_type="application/x-ndjson")
        else:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
                return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/admin/keys")
async def list_keys(master_key: str = Depends(verify_master_key), db = Depends(get_db)):
    keys = db.query(APIKey).all()
    return [{"name": k.name, "key_hash": k.key_hash, "created_at": k.created_at} for k in keys]

@app.post("/admin/keys")
async def create_key(req: CreateKeyRequest, master_key: str = Depends(verify_master_key), db = Depends(get_db)):
    new_key = f"ollama_{secrets.token_urlsafe(32)}"
    key_hash = hash_key(new_key)
    
    db_key = APIKey(
        key_hash=key_hash,
        name=req.name,
        created_at=int(time.time())
    )
    db.add(db_key)
    db.commit()
    
    return {
        "api_key": new_key,
        "name": req.name,
        "warning": "Save this key now. It will never be shown again."
    }

@app.delete("/admin/keys/{key_hash}")
async def revoke_key(key_hash: str, master_key: str = Depends(verify_master_key), db = Depends(get_db)):
    key_record = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
    if not key_record:
        raise HTTPException(status_code=404, detail="Key not found")
    db.delete(key_record)
    db.commit()
    return {"status": "revoked"}

# Static Content
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Ollama API Server</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #333; }
        pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
        code { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 0.9em; }
        .status { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
        .status-ok { background: #e6ffed; color: #22863a; }
        h1 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
        a { color: #0366d6; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Ollama API Server <span class="status status-ok">Active</span></h1>
    <p>This is a high-performance wrapper for Ollama, providing OpenAI-compatible endpoints and API key authentication.</p>
    
    <h3>Available Endpoints</h3>
    <ul>
        <li><code>GET /health</code> - Server health check</li>
        <li><code>GET /v1/models</code> - List available models</li>
        <li><code>POST /v1/chat/completions</code> - OpenAI-compatible chat</li>
        <li><code>POST /api/generate</code> - Ollama-native generation</li>
        <li><code>GET /api-docs</code> - API Documentation</li>
    </ul>

    <h3>Quick Start</h3>
    <pre><code>curl https://ollama-fastapi-railway-deployment-qst2ba.fly.dev/v1/chat/completions \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "tinyllama:latest",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'</code></pre>

    <p><a href="/docs">Swagger UI</a> | <a href="/api-docs">API Guide</a></p>
</body>
</html>
"""

API_DOCS_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>API Documentation - Ollama Server</title>
    <style>
        body { font-family: -apple-system, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #24292e; }
        h1, h2, h3 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
        pre { background: #f6f8fa; padding: 16px; border-radius: 6px; overflow: auto; }
        code { font-family: monospace; background: rgba(27,31,35,0.05); padding: 0.2em 0.4em; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #dfe2e5; padding: 8px 12px; text-align: left; }
        th { background: #f6f8fa; }
    </style>
</head>
<body>
    <h1>API Documentation</h1>
    
    <h2>Authentication</h2>
    <p>All API requests must include a Bearer token in the <code>Authorization</code> header:</p>
    <pre><code>Authorization: Bearer ollama_...</code></pre>

    <h2>Chat Completions (OpenAI Compatible)</h2>
    <p><code>POST /v1/chat/completions</code></p>
    <table>
        <tr><th>Field</th><th>Type</th><th>Description</th></tr>
        <tr><td>model</td><td>string</td><td>Model name (e.g., tinyllama:latest)</td></tr>
        <tr><td>messages</td><td>array</td><td>List of message objects</td></tr>
        <tr><td>stream</td><td>boolean</td><td>Enable streaming (SSE)</td></tr>
    </table>

    <h2>Admin API</h2>
    <p>Admin endpoints require the <code>X-Master-Key</code> header.</p>
    <ul>
        <li><code>GET /admin/keys</code> - List all keys</li>
        <li><code>POST /admin/keys</code> - Create a new key</li>
        <li><code>DELETE /admin/keys/{hash}</code> - Revoke a key</li>
    </ul>
</body>
</html>
"""

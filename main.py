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

# Configure logging
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

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is not set")

engine = None
SessionLocal = None
Base = declarative_base()

class APIKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String(64), unique=True, index=True)
    name = Column(String(100))
    created_at = Column(BigInteger, default=lambda: int(time.time()))

def init_db():
    global engine, SessionLocal
    if not DATABASE_URL:
        return
    try:
        # Neon PostgreSQL connection
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

@app.on_event("startup")
async def startup_event():
    init_db()

# Security
security = HTTPBearer(auto_error=False)

def get_db():
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security), db = Depends(get_db)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header. Use: Bearer YOUR_API_KEY")
    token = credentials.credentials
    key_hash = hash_key(token)
    try:
        db_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
        if not db_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return token
    except OperationalError:
        raise HTTPException(status_code=503, detail="Database connection error")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:0.5b")
MASTER_KEY = os.getenv("MASTER_KEY", "ollama-master-key-change-me")

async def verify_master_key(x_master_key: str = Header(None)):
    if not x_master_key:
        raise HTTPException(status_code=401, detail="Missing X-Master-Key header")
    if x_master_key != MASTER_KEY:
        raise HTTPException(status_code=403, detail="Invalid master key")
    return x_master_key

# ============ REQUEST MODELS ============

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
    rate_limit: Optional[int] = 1000

class RevokeKeyRequest(BaseModel):
    key_hash: str

# ============ PUBLIC ENDPOINTS ============

@app.get("/health")
async def health():
    ollama_ready = "not ready"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code == 200:
                ollama_ready = "connected"
    except:
        pass
    return {
        "status": "ok",
        "ollama": ollama_ready,
        "auth": "enabled",
        "database": SessionLocal is not None
    }

@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
@app.get("/UI", response_class=HTMLResponse)
async def web_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ollama API Server</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f0f23;color:#e0e0e0;min-height:100vh}
.container{max-width:900px;margin:0 auto;padding:40px 20px}
h1{font-size:2.5rem;margin-bottom:10px;background:linear-gradient(90deg,#00d4ff,#7b2cbf);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:#888;margin-bottom:40px}
.card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:24px;margin-bottom:20px}
.card h2{color:#00d4ff;margin-bottom:16px;font-size:1.2rem}
.endpoint{background:#0f0f1a;border-left:3px solid #00d4ff;padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0;font-family:"Courier New",monospace;font-size:.9rem}
.method{color:#7ee787;font-weight:bold;margin-right:8px}
.url{color:#dcdcaa}
.status{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.85rem;font-weight:600}
.status.ok{background:#1a472a;color:#7ee787}
.status.warn{background:#4a3a1a;color:#ffa500}
.chat-box{height:400px;overflow-y:auto;background:#0f0f1a;border-radius:8px;padding:16px;margin-bottom:16px}
.message{margin-bottom:12px;padding:12px;border-radius:8px;max-width:80%}
.message.user{background:#1a3a5c;margin-left:auto}
.message.assistant{background:#2a2a4a}
.input-row{display:flex;gap:10px}
input{flex:1;background:#0f0f1a;border:1px solid #2a2a4a;color:#e0e0e0;padding:12px;border-radius:8px;font-size:1rem}
button{background:linear-gradient(90deg,#00d4ff,#7b2cbf);color:white;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-weight:600;font-size:1rem}
button:hover{opacity:.9}
.code-block{background:#0f0f1a;border-radius:8px;padding:16px;overflow-x:auto;font-family:"Courier New",monospace;font-size:.85rem;color:#dcdcaa;margin:10px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.nav{display:flex;gap:10px;margin-bottom:30px;flex-wrap:wrap}
.nav a{color:#00d4ff;text-decoration:none;padding:8px 16px;border:1px solid #2a2a4a;border-radius:8px;font-size:.9rem}
.nav a:hover{background:#1a1a2e}
.key-input{width:100%;margin-bottom:10px}
.alert{background:#1a472a;border:1px solid #2a5a3a;color:#7ee787;padding:12px;border-radius:8px;margin-bottom:16px;display:none}
.alert.error{background:#4a1a1a;border-color:#5a2a2a;color:#ff6b6b}
.table{width:100%;border-collapse:collapse;margin:10px 0}
.table th,.table td{padding:10px;text-align:left;border-bottom:1px solid #2a2a4a;font-size:.9rem}
.table th{color:#00d4ff;font-weight:600}
.table td{color:#ccc}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
<div class="nav">
<a href="/ui">Dashboard</a>
<a href="/docs">Swagger UI</a>
<a href="/redoc">ReDoc</a>
</div>
<h1>Ollama API Server</h1>
<p class="subtitle">Self-hosted LLM with FastAPI + Ollama + API Keys (PostgreSQL)</p>

<div class="card">
<h2>API Key Required</h2>
<p style="color:#888;margin-bottom:12px">All API endpoints require a Bearer token. Enter your key below to test:</p>
<input type="password" id="apiKeyInput" class="key-input" placeholder="Enter your API key (ollama_xxxxxxxx...)" value="">
<div class="alert" id="keyAlert"></div>
</div>

<div class="card">
<h2>Connection Status</h2>
<div id="status">Checking...</div>
<div style="margin-top:10px;font-size:.9rem;color:#888">Base URL: <span id="baseUrl"></span></div>
</div>

<div class="card">
<h2>API Endpoints</h2>
<div class="endpoint"><span class="method">GET</span><span class="url">/health</span> - Health check</div>
<div class="endpoint"><span class="method">GET</span><span class="url">/v1/models</span> - List models</div>
<div class="endpoint"><span class="method">POST</span><span class="url">/v1/chat/completions</span> - Chat (OpenAI-compatible)</div>
<div class="endpoint"><span class="method">POST</span><span class="url">/api/generate</span> - Generate text</div>
<div class="endpoint"><span class="method">GET</span><span class="url">/api/models</span> - List models (native)</div>
<div class="endpoint"><span class="method">POST</span><span class="url">/api/pull</span> - Pull model</div>
</div>

<div class="card">
<h2>Test Chat</h2>
<div class="chat-box" id="chatBox"></div>
<div class="input-row">
<input type="text" id="chatInput" placeholder="Type a message..." onkeypress="if(event.key==='Enter')sendChat()">
<button onclick="sendChat()">Send</button>
</div>
</div>

<div class="card">
<h2>Key Management</h2>
<p style="color:#888;margin-bottom:10px">Use your MASTER_KEY in the X-Master-Key header to manage API keys.</p>
<div class="endpoint"><span class="method">POST</span><span class="url">/admin/keys</span> - Create new API key</div>
<div class="code-block">Headers: X-Master-Key: your-master-key<br>
Body: {"name": "my-app", "rate_limit": 1000}<br><br>
Response: {"api_key": "ollama_xxxxx", "warning": "Save this key now!"}</div>
</div>

<div class="grid">
<div class="card">
<h2>Python Example</h2>
<div class="code-block">
import requests<br><br>
url = "<span class='base-url'></span>/v1/chat/completions"<br>
headers = {<br>
&nbsp;&nbsp;"Content-Type": "application/json",<br>
&nbsp;&nbsp;"Authorization": "Bearer YOUR_API_KEY"<br>
}<br>
data = {<br>
&nbsp;&nbsp;"model": "qwen2.5:0.5b",<br>
&nbsp;&nbsp;"messages": [{"role": "user", "content": "Hello!"}]<br>
}<br><br>
res = requests.post(url, json=data, headers=headers)<br>
print(res.json())
</div>
</div>
<div class="card">
<h2>cURL Example</h2>
<div class="code-block">
curl -X POST <span class='base-url'></span>/v1/chat/completions<br>
-H "Content-Type: application/json"<br>
-H "Authorization: Bearer YOUR_API_KEY"<br>
-d '{"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"Hello!"}]}'
</div>
</div>
</div>

<div class="card">
<h2>Available Models</h2>
<table class="table">
<thead>
<tr><th>Model</th><th>Size</th><th>Status</th></tr>
</thead>
<tbody id="modelsTable">
<tr><td colspan="3" style="text-align:center">Loading models...</td></tr>
</tbody>
</table>
</div>

</div>
<script>
const baseUrl = window.location.origin;
document.getElementById('baseUrl').textContent = baseUrl;
document.querySelectorAll('.base-url').forEach(el=>el.textContent=baseUrl);
function getApiKey(){return document.getElementById('apiKeyInput').value.trim();}
function showAlert(msg,isError){
const el=document.getElementById('keyAlert');
el.textContent=msg;
el.style.display='block';
if(isError){el.classList.add('error');}else{el.classList.remove('error');}
}

async function checkHealth(){
try{
const res = await fetch(baseUrl + '/health');
const data = await res.json();
const el = document.getElementById('status');
if(data.ollama === 'connected'){
el.innerHTML = '<span class="status ok">Ollama Connected | Auth Enabled</span>';
}else{
el.innerHTML = '<span class="status warn">Ollama Starting...</span>';
}
updateModels();
}catch{
document.getElementById('status').innerHTML = '<span class="status warn">Server Starting...</span>';
}
}

async function updateModels(){
const key = getApiKey();
if(!key) return;
try{
const res = await fetch(baseUrl + '/v1/models', {
headers: {'Authorization': 'Bearer ' + key}
});
if(res.ok){
const data = await res.json();
const tbody = document.getElementById('modelsTable');
if(data.data && data.data.length > 0){
tbody.innerHTML = data.data.map(m => `<tr><td>${m.id}</td><td>-</td><td><span class="status ok">Ready</span></td></tr>`).join('');
} else {
tbody.innerHTML = '<tr><td colspan="3" style="text-align:center">No models available</td></tr>';
}
}
}catch(e){}
}

checkHealth();
setInterval(checkHealth, 5000);

async function sendChat(){
const key = getApiKey();
if(!key){showAlert("Please enter your API key above",true);return;}
const input = document.getElementById('chatInput');
const box = document.getElementById('chatBox');
const msg = input.value.trim();
if(!msg) return;
box.innerHTML += `<div class="message user">${msg}</div>`;
input.value = '';
box.scrollTop = box.scrollHeight;
try{
const res = await fetch(baseUrl + '/v1/chat/completions', {
method: 'POST',
headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key},
body: JSON.stringify({model: 'qwen2.5:0.5b', messages: [{role: 'user', content: msg}]})
});
if(res.status===401){showAlert("Invalid API key!",true);return;}
const data = await res.json();
const reply = data.choices?.[0]?.message?.content || 'No response';
box.innerHTML += `<div class="message assistant">${reply}</div>`;
box.scrollTop = box.scrollHeight;
showAlert("Message sent successfully",false);
}catch(e){
box.innerHTML += `<div class="message assistant" style="color:#ff6b6b">Error: ${e.message}</div>`;
}
}
</script>
</body>
</html>"""

# ============ KEY MANAGEMENT (MASTER KEY PROTECTED) ============

@app.post("/admin/keys")
async def create_key(req: CreateKeyRequest, master: str = Depends(verify_master_key), db = Depends(get_db)):
    raw_key = "ollama_" + secrets.token_urlsafe(32)
    key_hash = hash_key(raw_key)
    db_key = APIKey(key_hash=key_hash, name=req.name)
    db.add(db_key)
    db.commit()
    db.refresh(db_key)
    return {
        "api_key": raw_key,
        "name": req.name,
        "warning": "Save this key now - it will not be shown again!"
    }

@app.get("/admin/keys")
async def list_keys(master: str = Depends(verify_master_key), db = Depends(get_db)):
    keys = db.query(APIKey).all()
    return {"keys": [{"id": k.id, "name": k.name, "key_hash": k.key_hash, "created_at": k.created_at} for k in keys]}

@app.post("/admin/keys/revoke")
async def revoke_key(req: RevokeKeyRequest, master: str = Depends(verify_master_key), db = Depends(get_db)):
    db_key = db.query(APIKey).filter(APIKey.key_hash == req.key_hash).first()
    if db_key:
        db.delete(db_key)
        db.commit()
        return {"status": "revoked"}
    raise HTTPException(status_code=404, detail="Key not found")

# ============ PROTECTED API ENDPOINTS ============

@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=30)
            data = r.json()
            models = []
            for m in data.get("models", []):
                models.append({
                    "id": m["name"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama"
                })
            return {"object": "list", "data": models}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    model = req.model or DEFAULT_MODEL
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
        
        async def streamer(request_payload):
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=request_payload) as response:
                    async for line in response.aiter_lines():
                        if line:
                            yield line + "\n"

        if req.stream:
            return StreamingResponse(streamer(payload), media_type="text/event-stream")

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
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

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
        
        async def streamer(request_payload):
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=request_payload) as response:
                    async for line in response.aiter_lines():
                        if line:
                            yield line + "\n"

        if req.stream:
            return StreamingResponse(streamer(payload), media_type="application/x-ndjson")

        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.get("/api/models")
async def ollama_models(api_key: str = Depends(verify_api_key)):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=30)
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.post("/api/pull")
async def pull_model(req: PullModelRequest, api_key: str = Depends(verify_api_key)):
    try:
        payload = {"name": req.name, "stream": False}
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/pull", json=payload)
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

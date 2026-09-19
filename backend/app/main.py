import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.logging import logger
from app.db.database import init_db
from app.api.router import api_router

# Initialize Database tables
init_db()

app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise Document Intelligence & RAG Assistant with Hybrid Retrieval, Citations, and Evaluation Framework.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router, prefix="/api")

# Static Frontend mounting
frontend_path = settings.BASE_DIR / "frontend"
if not frontend_path.exists():
    frontend_path = settings.BASE_DIR / "backend" / "app" / "static"

if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.get("/", include_in_schema=False)
def read_root():
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Enterprise Document Intelligence API running. Access /docs for OpenAPI specifications."}



@app.on_event("startup")
def startup_event():
    logger.info(f"Starting {settings.APP_NAME} in environment '{settings.APP_ENV}'")
    logger.info(f"Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"LLM Provider: {'OpenAI REST API' if settings.LLM_API_KEY else 'Offline Mock Engine'}")

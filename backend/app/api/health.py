from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.database import get_db
from app.core.config import settings
from app.db.models import Document, Chunk

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    """Health status and database/vector index summary."""
    try:
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"

    doc_count = db.query(Document).count()
    chunk_count = db.query(Chunk).count()

    provider_name = "Offline Mock Engine"
    if settings.LLM_PROVIDER.lower() == "ollama":
        provider_name = f"Ollama Local ({settings.OLLAMA_MODEL})"
    elif settings.LLM_API_KEY:
        provider_name = f"OpenAI REST API ({settings.LLM_MODEL})"
    else:
        # Check if Ollama is accessible
        try:
            with httpx.Client(timeout=1.0) as client:
                r = client.get(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags")
                if r.status_code == 200:
                    models = [m.get("name", "") for m in r.json().get("models", [])]
                    if models:
                        provider_name = f"Ollama Local ({models[0]})"
        except Exception:
            pass

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "environment": settings.APP_ENV,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
        "llm_provider": provider_name,
        "documents_count": doc_count,
        "chunks_count": chunk_count
    }


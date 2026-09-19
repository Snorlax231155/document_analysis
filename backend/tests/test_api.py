from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "Enterprise Document Intelligence" in response.text


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "embedding_model" in data


def test_list_documents_endpoint():
    response = client.get("/api/documents")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_query_endpoint():
    payload = {
        "question": "What is the annual leave policy?",
        "top_k": 5,
        "enable_reranking": True
    }
    response = client.post("/api/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "retrieval_metadata" in data
    assert "performance" in data


def test_evaluation_run_endpoint():
    response = client.post("/api/evaluation/run")
    assert response.status_code == 200
    data = response.json()
    assert "recall_at_1" in data
    assert "mrr" in data
    assert "faithfulness" in data

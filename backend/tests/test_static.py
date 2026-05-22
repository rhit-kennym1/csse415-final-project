from fastapi.testclient import TestClient

from backend.main import app


def test_root_serves_frontend_index():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Phishing Website Detector" in resp.text


def test_static_css_served():
    client = TestClient(app)
    resp = client.get("/style.css")
    assert resp.status_code == 200
    assert "text/css" in resp.headers["content-type"]


def test_api_routes_take_precedence_over_static():
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

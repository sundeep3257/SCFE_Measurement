"""Lightweight Flask smoke tests (no ML models required)."""

import io

import pytest

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hemipelvis Radiograph Analysis" in response.data


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_reject_non_png_upload(client):
    data = {
        "image": (io.BytesIO(b"not a real image"), "test.jpg"),
    }
    response = client.post("/analyze", data=data, content_type="multipart/form-data")
    assert response.status_code == 302
    follow = client.get("/")
    assert b"Only PNG files are supported" in follow.data

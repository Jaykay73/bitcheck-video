import asyncio

import httpx

from app.config import settings
from main import app


def test_valid_output_file_served():
    path = settings.outputs_dir / "test_output_serving.jpg"
    path.write_bytes(b"fake-jpeg")

    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(f"/outputs/{path.name}")

    try:
        response = asyncio.run(run_request())
    finally:
        path.unlink(missing_ok=True)

    assert response.status_code == 200
    assert response.content == b"fake-jpeg"


def test_output_path_traversal_rejected():
    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/outputs/..%2FREADME.md")

    response = asyncio.run(run_request())

    assert response.status_code == 404


def test_missing_output_file_returns_404():
    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/outputs/not-here.jpg")

    response = asyncio.run(run_request())

    assert response.status_code == 404

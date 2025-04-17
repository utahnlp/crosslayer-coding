import pytest
import pytest_asyncio
from httpx import AsyncClient
import sys
import os
from typing import AsyncGenerator

# Ensure the project root is in the Python path for imports
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the FastAPI app instance
# Adjust the import path if your structure differs
try:
    from clt_server.main import app
except ImportError as e:
    print(f"Error importing FastAPI app: {e}")
    print(
        "Ensure the test is run from the project root or PYTHONPATH is set correctly."
    )
    # Optionally re-raise or exit if the app cannot be imported
    raise

# --- Fixtures ---


@pytest_asyncio.fixture(scope="function")  # Use function scope for client isolation
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Provides an httpx AsyncClient configured for the test app."""
    # Use the context manager for proper startup/shutdown event handling
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# --- Test Cases ---


@pytest.mark.asyncio
async def test_health_check_status_code(async_client: AsyncClient):
    """Tests if the health check endpoint returns a 200 OK status."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_check_response_body(async_client: AsyncClient):
    """Tests if the health check endpoint returns the correct JSON body."""
    response = await async_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

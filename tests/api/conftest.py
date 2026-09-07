import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from tests.support.repository import build_service


@pytest.fixture(scope="session")
def service():
    return build_service()


@pytest.fixture
def client(service):
    with TestClient(create_app(service=service)) as client:
        yield client

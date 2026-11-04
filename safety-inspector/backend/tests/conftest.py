import importlib
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    import os

    folder = tmp_path_factory.mktemp("database")
    os.environ["DATABASE_URL"] = f"sqlite:///{folder / 'test.db'}"
    os.environ["DATA_DIR"] = str(folder)
    os.environ["DEMO_AUTOSTART"] = "false"
    os.environ["DEMO_SEED_INCIDENTS"] = "24"
    os.environ["DEMO_SEED_MINUTES"] = "60"
    os.environ["JWT_SECRET"] = "test-secret-with-at-least-thirty-two-characters"
    command.upgrade(Config("alembic.ini"), "head")
    main = importlib.import_module("app.main")
    with TestClient(main.app) as test_client:
        yield test_client

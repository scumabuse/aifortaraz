import importlib
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

TEST_AI_KEY = "test-ai-module-key-from-environment"


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
    # The seed reads these; fixed values keep the suite independent of the shell.
    os.environ["DEMO_MODE"] = "true"
    os.environ["AI_CAMERAS"] = ""
    os.environ["AI_MJPEG_BASE_URL"] = "http://localhost:8001"
    os.environ["SEED_AI_API_KEY"] = TEST_AI_KEY
    command.upgrade(Config("alembic.ini"), "head")
    seed = importlib.import_module("app.seed")
    # Never write the generated AI key into the real project directory.
    seed.AI_KEY_FILE = folder / ".secrets" / "ai_api_key"
    main = importlib.import_module("app.main")
    with TestClient(main.app) as test_client:
        yield test_client

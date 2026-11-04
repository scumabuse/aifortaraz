"""Run the local backend and frontend; terminate both on Ctrl+C."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
os.environ["DATA_DIR"] = str((ROOT / os.getenv("DATA_DIR", "./data")).resolve())
if os.getenv("DATABASE_URL", "").startswith("sqlite:///./"):
    os.environ["DATABASE_URL"] = (
        "sqlite:///" + (ROOT / os.environ["DATABASE_URL"][12:]).as_posix()
    )


def main():
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT / "backend",
        check=True,
    )
    processes = []
    try:
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                ],
                cwd=ROOT / "backend",
            )
        )
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("Node.js and npm are required")
        processes.append(subprocess.Popen([npm, "run", "dev"], cwd=ROOT / "frontend"))
        while all(p.poll() is None for p in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    process.terminate()
        for process in processes:
            process.wait()


if __name__ == "__main__":
    main()

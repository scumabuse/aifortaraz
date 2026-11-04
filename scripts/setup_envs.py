"""`make setup`: create the three environments of the product (idempotent).

  .venv (root)                           integration-tests/requirements.txt + Playwright Chromium
  aifortaraz/safety-inspector/.venv      backend/requirements.txt; frontend node_modules via npm ci
  safety-ai/.venv                        pip install -e ".[dev,onnx,openvino]" (CPU torch wheels)

Existing environments are kept; only missing pieces are installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import AI, FRONTEND, ROOT, WEBSITE, npm_command, venv_python  # noqa: E402


def run(args: list[str], cwd: Path) -> None:
    """Run a setup step, echoing it."""
    print(f"$ {' '.join(args)}   (in {cwd})", flush=True)
    subprocess.run(args, cwd=cwd, check=True)  # noqa: S603


def ensure_venv(project: Path) -> Path:
    """Create ``project/.venv`` when missing and return its interpreter."""
    py = venv_python(project)
    if not py.exists():
        run([sys.executable, "-m", "venv", ".venv"], cwd=project)
        run([str(venv_python(project)), "-m", "pip", "install", "-q", "-U", "pip"], cwd=project)
    return venv_python(project)


def main() -> int:
    """Create whatever is missing."""
    root_py = ensure_venv(ROOT)
    run([str(root_py), "-m", "pip", "install", "-q", "-r", "integration-tests/requirements.txt"], cwd=ROOT)
    run([str(root_py), "-m", "playwright", "install", "chromium"], cwd=ROOT)
    web_py = ensure_venv(WEBSITE)
    run([str(web_py), "-m", "pip", "install", "-q", "-r", "backend/requirements.txt"], cwd=WEBSITE)
    if not (FRONTEND / "node_modules").exists():
        run([npm_command(), "ci"], cwd=FRONTEND)
    ai_py = ensure_venv(AI)
    probe = subprocess.run([str(ai_py), "-c", "import safety_ai, ultralytics"], cwd=AI, check=False,  # noqa: S603
                           capture_output=True)
    if probe.returncode != 0:
        run([str(ai_py), "-m", "pip", "install", "-q", "torch", "torchvision", "--index-url",
             "https://download.pytorch.org/whl/cpu"], cwd=AI)
        run([str(ai_py), "-m", "pip", "install", "-q", "-e", ".[dev,onnx,openvino]"], cwd=AI)
    print("setup: done — next `make init`, `make doctor`, `make dev`")
    return 0


if __name__ == "__main__":
    sys.exit(main())

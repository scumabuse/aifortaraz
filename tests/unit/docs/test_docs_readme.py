"""README.md tells how to run the module together with the website (integration root)."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_readme_points_to_the_integration_root_and_lists_the_ports() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Running with the other half", 1)[1]
    assert "[README](../README.md)" in section
    assert "[contract/CONTRACT.md](../contract/CONTRACT.md)" in section
    for port, service in (("8001", "MJPEG"), ("9100", "health"), ("8010", "mock backend")):
        row = next(line for line in section.splitlines() if line.startswith(f"| {port} |"))
        assert service in row, row
    assert "python -m tools.mock_backend --port 8010" in section

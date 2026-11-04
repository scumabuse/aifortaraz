"""Contract and schema tests (integration spec 8.1).

- the contract text is byte-identical in the root and in both repositories;
- the canonical examples (contract/examples/*.json) validate with BOTH sides' models;
- the website's incoming models and the AI's outgoing models agree on property names, required
  fields, enums and types for every message (and the other way round for the config response);
- the real backend rejects ten malformed events with 422 + field errors and stays healthy.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from conftest import ROOT, Backend

WEBSITE = ROOT / "aifortaraz" / "safety-inspector"
AI = ROOT / "safety-ai"
EXAMPLES = ROOT / "contract" / "examples"
TOOL = Path(__file__).resolve().parent / "helpers" / "schema_tool.py"
PYTHONS = {
    "website": WEBSITE / ".venv" / "Scripts" / "python.exe" if (WEBSITE / ".venv" / "Scripts").exists() else WEBSITE / ".venv" / "bin" / "python",
    "ai": AI / ".venv" / "Scripts" / "python.exe" if (AI / ".venv" / "Scripts").exists() else AI / ".venv" / "bin" / "python",
}
# AI -> site messages (the AI produces, the website consumes); the config goes the other way.
PRODUCER_IS_AI = ("event", "patch", "heartbeat", "stats", "ws_detections", "ws_frame", "ws_heartbeat",
                  "details.ppe_violation", "details.zone_intrusion", "details.vehicle_proximity",
                  "details.fall_detected", "details.person_immobile", "details.overcrowding", "details.custom")


def run_tool(side: str, command: str, *rest: str) -> Any:
    """Run ``schema_tool.py <command> <side> <rest…>`` inside the side's venv and parse its JSON output."""
    done = subprocess.run([str(PYTHONS[side]), str(TOOL), command, side, *rest], capture_output=True, text=True,  # noqa: S603
                          encoding="utf-8", timeout=120, check=False)
    assert done.returncode == 0, f"schema tool failed for {side}: {done.stderr[-2000:]}"
    return json.loads(done.stdout)


# --------------------------------------------------------------------------------------------
# Contract text
# --------------------------------------------------------------------------------------------


def test_contract_text_in_sync() -> None:
    """contract/CONTRACT.md and both repositories' docs/CONTRACT.md are byte-identical (spec 4)."""
    paths = [ROOT / "contract" / "CONTRACT.md", WEBSITE / "docs" / "CONTRACT.md", AI / "docs" / "CONTRACT.md"]
    digests = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert len(set(digests.values())) == 1, f"contract copies differ: {digests}"


# --------------------------------------------------------------------------------------------
# Examples and schema parity
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["website", "ai"])
def test_canonical_examples_validate(side: str) -> None:
    """Every contract/examples/*.json is accepted by the side's model of that message."""
    results = run_tool(side, "validate", str(EXAMPLES))
    assert len(results) >= 14, f"examples missing: {sorted(results)}"
    failing = {name: error for name, error in results.items() if error != "ok"}
    assert not failing, f"{side} rejects canonical examples: {json.dumps(failing, indent=2, ensure_ascii=False)}"


def _resolve(node: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    while "$ref" in node:
        name = node["$ref"].split("/")[-1]
        node = {**root.get("$defs", {})[name], **{k: v for k, v in node.items() if k != "$ref"}}
    return node


def _variants(node: dict[str, Any], root: dict[str, Any]) -> list[dict[str, Any]]:
    node = _resolve(node, root)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in node:
            out: list[dict[str, Any]] = []
            for item in node[key]:
                out.extend(_variants(item, root))
            return out
    return [node]


def flatten(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """path -> {"types": set, "enum": set | None, "required": bool, "free": bool}.

    Object properties become ``a.b``, array items ``a[]``, map values ``a{}``. Union variants are
    merged: types and enums are united, a property is required only when every object variant
    requires it. ``free`` marks objects that accept arbitrary keys (free-form ``extra``, maps).
    """
    out: dict[str, dict[str, Any]] = {}

    def visit(node: dict[str, Any], path: str, required: bool) -> None:
        variants = _variants(node, schema)
        entry = out.setdefault(path, {"types": set(), "enum": None, "required": required, "free": False})
        entry["required"] = entry["required"] and required
        object_variants = []
        for variant in variants:
            kind = variant.get("type")
            if "const" in variant:
                entry["enum"] = (entry["enum"] or set()) | {json.dumps(variant["const"])}
                entry["types"].add(type(variant["const"]).__name__.replace("str", "string").replace("bool", "boolean"))
            if "enum" in variant:
                entry["enum"] = (entry["enum"] or set()) | {json.dumps(v) for v in variant["enum"]}
            if isinstance(kind, list):
                entry["types"].update(kind)
            elif kind:
                entry["types"].add(kind)
            if kind == "object" or "properties" in variant:
                object_variants.append(variant)
            if kind == "array" and "items" in variant:
                visit(variant["items"], f"{path}[]", True)
            if kind == "array" and "prefixItems" in variant:
                for item in variant["prefixItems"]:
                    visit(item, f"{path}[]", True)
        for variant in object_variants:
            props = variant.get("properties", {})
            extra = variant.get("additionalProperties")
            if extra not in (None, False) and not props:
                entry["free"] = True
                if isinstance(extra, dict) and extra:
                    visit(extra, f"{path}{{}}", True)
            elif extra not in (None, False):
                entry["free"] = True
            required_here = set(variant.get("required", []))
            for name, sub in props.items():
                every = all(name in set(v.get("required", [])) for v in object_variants if name in v.get("properties", {}))
                visit(sub, f"{path}.{name}" if path else name, required and name in required_here and every)

    visit(schema, "", True)
    return out


def _compatible(produced: set[str], accepted: set[str]) -> bool:
    accepted = set(accepted) | ({"integer"} if "number" in accepted else set())
    return not accepted or produced - {"null"} <= accepted | ({"null"} if "null" in accepted else set())


def parity_problems(message: str, producer: dict[str, Any], consumer: dict[str, Any]) -> list[str]:
    """Differences between what the producer sends and what the consumer accepts."""
    prod, cons = flatten(producer), flatten(consumer)
    problems: list[str] = []
    for path, p in prod.items():
        if not path or (message == "ws_heartbeat" and path == "type"):
            continue
        if path not in cons:
            parent = path.rsplit(".", 1)[0] if "." in path else ""
            free_parent = any(cons.get(prefix, {}).get("free") for prefix in _prefixes(path))
            if not free_parent and parent != path:
                problems.append(f"{message}: producer sends `{path}` but the consumer does not accept it")
            continue
        c = cons[path]
        if not _compatible(p["types"], c["types"]):
            problems.append(f"{message}: `{path}` type {sorted(p['types'])} not accepted by {sorted(c['types'])}")
        if c["enum"] is not None and p["enum"] is not None and not p["enum"] <= c["enum"]:
            problems.append(f"{message}: `{path}` values {sorted(p['enum'] - c['enum'])} not accepted")
        if "null" in p["types"] and "null" not in c["types"] and c["types"]:
            problems.append(f"{message}: `{path}` may be null but the consumer rejects null")
    for path, c in cons.items():
        if not path or not c["required"] or (message == "ws_heartbeat" and path == "type"):
            continue
        if path not in prod and not any(prod.get(prefix, {}).get("free") for prefix in _prefixes(path)):
            problems.append(f"{message}: consumer requires `{path}` but the producer never sends it")
        elif path in prod and not prod[path]["required"] and not _optional_ok(message, path):
            problems.append(f"{message}: consumer requires `{path}` but the producer may omit it")
    return problems


def _prefixes(path: str) -> list[str]:
    parts = path.replace("[]", ".[]").replace("{}", ".{}").split(".")
    out = []
    for i in range(1, len(parts)):
        out.append(".".join(parts[:i]).replace(".[]", "[]").replace(".{}", "{}"))
    return out


def _optional_ok(message: str, path: str) -> bool:
    # Fields both sides treat as optional extras of the producer's model.
    return False


def test_schema_parity() -> None:
    """Website incoming models == AI outgoing models (and the config the other way round)."""
    website, ai = run_tool("website", "dump"), run_tool("ai", "dump")
    assert set(website) == set(ai), f"message sets differ: {set(website) ^ set(ai)}"
    problems: list[str] = []
    for message in sorted(website):
        if message in PRODUCER_IS_AI:
            problems += parity_problems(message, ai[message], website[message])
        else:
            problems += parity_problems(message, website[message], ai[message])
    schemas_dir = ROOT / "contract" / "schemas"
    schemas_dir.mkdir(exist_ok=True)
    for side, data in (("website", website), ("ai", ai)):
        for message, schema in data.items():
            (schemas_dir / f"{side}.{message}.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
                                                               encoding="utf-8")
    assert not problems, "schema parity problems:\n  " + "\n  ".join(problems)


# --------------------------------------------------------------------------------------------
# Real backend rejects malformed input
# --------------------------------------------------------------------------------------------


def _jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 180), (120, 90, 60)).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _event(zone_id: str, track: int = 41, **over: Any) -> dict[str, Any]:
    payload = json.loads((EXAMPLES / "event_ppe_violation.json").read_text(encoding="utf-8"))
    payload.update({"camera_id": "cam-1", "zone_id": zone_id, "track_id": track,
                    "external_id": f"cam-1-1793715667-{track}-ppe_violation"})
    payload["details"]["boxes"][0]["track_id"] = track
    payload.update(over)
    return payload


def test_real_backend_rejects_bad_payloads(shared_backend: Backend) -> None:
    """Ten malformed events → 422 with field errors; the backend never crashes (spec 4.8)."""
    backend = shared_backend
    config = backend.client.get("/api/v1/ingest/config/cam-1", headers=backend.ai)
    assert config.status_code == 200, f"config of cam-1: {config.status_code} {config.text[:300]}"
    zone = config.json()["zones"][0]["id"]
    snap = _jpeg()
    good = _event(zone)
    cases: list[tuple[str, dict[str, Any], bytes, str]] = [
        ("bad type enum", _event(zone, 42, type="smoking", external_id="cam-1-1793715667-42-smoking"), snap, "type"),
        ("bad severity enum", _event(zone, 43, severity="extreme"), snap, "severity"),
        ("bbox > 1", _event(zone, 44, details={**good["details"], "boxes": [{"track_id": 44, "label": "person", "bbox": [0.9, 0.2, 0.3, 0.4], "conf": 0.9}]}), snap, "bbox"),
        ("missing frame_size", _event(zone, 45, details={k: v for k, v in good["details"].items() if k != "frame_size"}), snap, "frame_size"),
        ("bad external_id", _event(zone, 46, external_id="not-a-contract-id"), snap, "external_id"),
        ("unknown box label", _event(zone, 47, details={**good["details"], "boxes": [{"track_id": 47, "label": "dinosaur", "bbox": [0.1, 0.2, 0.3, 0.4], "conf": 0.9}]}), snap, "label"),
        ("confidence > 1", _event(zone, 48, confidence=1.7), snap, "confidence"),
        ("missing details key", _event(zone, 49, details={k: v for k, v in good["details"].items() if k != "missing_ppe"}), snap, "missing_ppe"),
        ("oversized snapshot", _event(zone, 50), b"\xff\xd8\xff" + b"0" * (5 * 1024 * 1024 + 10), "snapshot"),
        ("non-JPEG snapshot", _event(zone, 51), b"\x89PNG\r\n\x1a\n" + b"0" * 64, "snapshot"),
    ]
    results = []
    for title, payload, snapshot, field_hint in cases:
        response = backend.client.post("/api/v1/ingest/events", headers=backend.ai, data={"payload": json.dumps(payload)},
                                       files={"snapshot": ("s.jpg", snapshot, "image/jpeg")})
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        results.append((title, response.status_code, json.dumps(body, ensure_ascii=False)[:300]))
        expected = (413, 422) if title == "oversized snapshot" else (422,)
        assert response.status_code in expected, f"{title}: expected {expected}, got {response.status_code} {response.text[:300]}"
        assert "error" in body, f"{title}: error body must be {{'error': ...}} (contract 4), got {body}"
        assert field_hint in json.dumps(body), f"{title}: the error must name the field `{field_hint}`: {body}"
    health = backend.client.get("/health")
    assert health.status_code == 200, "backend must stay healthy after malformed input (contract 4.8)"
    ok = backend.client.post("/api/v1/ingest/events", headers=backend.ai,
                             data={"payload": json.dumps(_event(zone, 60))}, files={"snapshot": ("s.jpg", snap, "image/jpeg")})
    assert ok.status_code == 201, f"a valid event after the malformed ones must still be accepted: {ok.text[:300]}"

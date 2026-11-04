# safety-ai developer targets (spec 7.1, 8.7).
#
# GNU make on Linux, macOS and Windows (ezwinports make). On Windows make runs recipes with sh.exe
# when it is on PATH (Git Bash) and with cmd.exe otherwise, so every recipe only invokes
# "$(PY) -m <module>" or "$(PY) <script>": no shell builtins, no && chains, no rm -rf.
#
#   make install                      editable install with the dev, onnx, openvino, data extras
#   make check                        lint, typecheck, test, e2e, placeholders (in that order)
#   make eval MODEL=ppe_n_latest      pass a registry name; ARGS="..." adds CLI options

ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
BOOTSTRAP_PY ?= python
else
PY ?= .venv/bin/python
BOOTSTRAP_PY ?= python3
endif

COVERAGE_JSON := reports/coverage.json
COVERAGE_MIN := 90
MODEL ?= ppe_s_latest
ARGS ?=
# The default interpreter path becomes a prerequisite, so a missing .venv is created first.
VENV_PY := $(filter .venv/%,$(PY))
empty :=
indent := $(empty)  $(empty)

.DEFAULT_GOAL := help
.NOTPARALLEL:
.PHONY: help venv install lint format typecheck test e2e check placeholders \
	demo mock train eval bench models selftest lock

help:
	$(info safety-ai make targets:)
	$(info $(indent)venv          create .venv with $(BOOTSTRAP_PY) when it does not exist)
	$(info $(indent)install       pip install -e ".[dev,onnx,openvino,data]" (creates .venv first))
	$(info $(indent)lint          ruff check and ruff format --check)
	$(info $(indent)format        sort imports and reformat with ruff)
	$(info $(indent)typecheck     mypy safety_ai tools)
	$(info $(indent)test          unit, property, scenario and integration tests with coverage, then)
	$(info $(indent)              >= $(COVERAGE_MIN) % line coverage of safety_ai/logic and safety_ai/backend)
	$(info $(indent)e2e           end-to-end tests with the real models (tests/e2e, marker e2e))
	$(info $(indent)check         lint, typecheck, test, e2e and placeholders, in that order)
	$(info $(indent)placeholders  fail on placeholders in safety_ai, tools and tests)
	$(info $(indent)demo          offline demo on data/samples/synthetic_people.mp4 with MJPEG)
	$(info $(indent)mock          strict mock backend on port 8000)
	$(info $(indent)train, eval, bench, selftest  safety-ai CLI commands (MODEL=$(MODEL), ARGS=...))
	$(info $(indent)models        download or export the weights listed in models/registry.json)
	$(info $(indent)lock          regenerate requirements.lock for the CPU Docker image (uv))
	@echo Interpreter: $(PY)

.venv/Scripts/python.exe .venv/bin/python:
	$(BOOTSTRAP_PY) -m venv --upgrade-deps .venv

venv: $(VENV_PY)

install: $(VENV_PY)
	$(PY) -m pip install -e ".[dev,onnx,openvino,data]"

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff check --select I --fix .
	$(PY) -m ruff format .

typecheck:
	$(PY) -m mypy safety_ai tools

test:
	$(PY) -m pytest tests/unit tests/property tests/scenarios tests/integration --cov=safety_ai --cov-report=json:$(COVERAGE_JSON) --cov-report=term
	$(PY) tools/check_coverage.py --report $(COVERAGE_JSON) --package safety_ai/logic --package safety_ai/backend --min $(COVERAGE_MIN)

e2e:
	$(PY) -m pytest tests/e2e -m "e2e"

check: lint typecheck test e2e placeholders

placeholders:
	$(PY) tools/check_placeholders.py

demo:
	$(PY) -m safety_ai.cli demo --source data/samples/synthetic_people.mp4 --zones tools/demo_zones.yaml --mjpeg --loop

mock:
	$(PY) -m safety_ai.cli mock-backend --port 8000 --config tools/mock_config.yaml

train:
	$(PY) -m safety_ai.cli train $(ARGS)

eval:
	$(PY) -m safety_ai.cli eval --model $(MODEL) $(ARGS)

bench:
	$(PY) -m safety_ai.cli bench $(ARGS)

selftest:
	$(PY) -m safety_ai.cli selftest $(ARGS)

models:
	$(PY) -m safety_ai.inference.models_setup

# requirements.lock pins the CPU Docker image (linux x86_64, Python 3.11). torch and torchvision
# resolve to the +cpu wheels of the PyTorch index, so installing the lock needs
# --extra-index-url https://download.pytorch.org/whl/cpu. tools/lock_overrides.txt removes
# opencv-python (the GUI build ultralytics depends on); only opencv-python-headless is pinned.
lock:
	$(PY) -m uv pip compile pyproject.toml --extra onnx --extra openvino --extra data --python-version 3.11 --python-platform x86_64-manylinux_2_28 --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --override tools/lock_overrides.txt -o requirements.lock

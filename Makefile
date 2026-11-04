# Digital Safety Inspector — website + AI module as one product (integration spec 7.6).
#
# GNU make on Windows (ezwinports make), Linux and macOS. Recipes only call "$(PY) <script>",
# "npm" and "docker compose", so they behave the same under sh and cmd.
#
#   make setup       create the three virtual environments / node_modules (first run)
#   make init        create .env (AI key, JWT secret) and copy it into both repositories
#   make doctor      green/red pre-flight table
#   make dev         backend + frontend + AI on this machine; Ctrl+C stops all three
#   make up          the same with docker compose (CPU image of the AI module)
#   make e2e         root integration suite (contract, schema parity, flows, chaos)
#   make e2e-full    e2e + real-model smoke with screenshots + performance budget

ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
WEB_PY := aifortaraz/safety-inspector/.venv/Scripts/python.exe
AI_PY := safety-ai/.venv/Scripts/python.exe
BOOT_PY ?= python
else
PY ?= .venv/bin/python
WEB_PY := aifortaraz/safety-inspector/.venv/bin/python
AI_PY := safety-ai/.venv/bin/python
BOOT_PY ?= python3
endif

COMPOSE := docker compose -p inspector
PYTEST_ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help setup init doctor dev restart-ai up down logs ps e2e e2e-full demo-preset videos \
	check-repos verify

help:
	$(info Digital Safety Inspector — integration root targets:)
	$(info   setup        create .venv (root), aifortaraz/safety-inspector/.venv + node_modules, safety-ai/.venv)
	$(info   init         create/complete .env and copy it into both repositories)
	$(info   doctor       pre-flight checks (ports, .env, AI key, weights, videos, versions))
	$(info   dev          run backend :8000, frontend :5173, AI :8001/:9100 (Ctrl+C stops all))
	$(info   restart-ai   restart only the AI module of a running `make dev`)
	$(info   up / down / logs / ps   docker compose equivalents (needs Docker))
	$(info   e2e          root integration suite 8.1-8.2 (<= 15 min))
	$(info   e2e-full     e2e + real-model smoke, screenshots and performance budget)
	$(info   demo-preset  presentation preset against the running website (idempotent))
	$(info   videos       build the demo videos safety-ai/data/samples/cam1.mp4, cam2.mp4)
	$(info   check-repos  `make check` in both repositories)
	$(info   verify       scripted manual-equivalent pass (Playwright))
	@$(BOOT_PY) -c "pass"

setup:
	$(BOOT_PY) scripts/setup_envs.py

init:
	$(BOOT_PY) scripts/init_env.py

doctor:
	$(PY) scripts/doctor.py

dev:
	$(PY) scripts/run_all.py

restart-ai:
	$(PY) scripts/run_all.py --restart-ai

up:
	$(COMPOSE) up -d --build
	$(COMPOSE) ps

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail 200

ps:
	$(COMPOSE) ps

e2e:
	$(PY) -m pytest integration-tests -m "not slow" $(PYTEST_ARGS)

e2e-full:
	$(PY) -m pytest integration-tests $(PYTEST_ARGS)

demo-preset:
	$(PY) scripts/demo_preset.py

videos:
	$(AI_PY) safety-ai/tools/make_demo_videos.py

check-repos:
	$(MAKE) -C safety-ai check
	$(MAKE) -C aifortaraz/safety-inspector check

verify:
	$(PY) scripts/verify_run.py

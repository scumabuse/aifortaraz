.PHONY: dev test check

ifeq ($(OS),Windows_NT)
PY ?= .venv/Scripts/python.exe
else
PY ?= .venv/bin/python
endif

# Recipes run under sh or cmd. The backend tests run inside backend/, so a relative
# interpreter path is anchored at this directory; absolute paths and bare commands
# (found on PATH) are used as given.
PY_ANCHORED := $(PY)
ifneq ($(findstring /,$(PY)),)
ifeq ($(filter /% %:,$(firstword $(subst :,: ,$(PY)))),)
PY_ANCHORED := $(CURDIR)/$(PY)
endif
endif

dev:
	"$(PY_ANCHORED)" scripts/dev.py

test:
	cd backend && "$(PY_ANCHORED)" -m pytest

check:
	cd backend && "$(PY_ANCHORED)" -m pytest -q -p no:cacheprovider
	"$(PY_ANCHORED)" -m ruff check backend scripts
	npm --prefix frontend test
	npm --prefix frontend run lint
	npm --prefix frontend run build

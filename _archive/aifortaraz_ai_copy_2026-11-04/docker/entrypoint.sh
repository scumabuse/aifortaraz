#!/bin/sh
# Container entrypoint: make sure the registered weights exist (spec 6.7, 7.4: "weights are
# downloaded or exported at first start by make models"), then hand over to the safety-ai CLI.
set -eu

cd /app

case "${1:-}" in
  run|demo|selftest|bench|eval|export|autolabel)
    if ! python -m safety_ai.inference.models_setup --check >/dev/null 2>&1; then
      echo "safety-ai: registered weights missing or invalid, running 'make models'"
      make models
    fi
    ;;
esac

exec safety-ai "$@"

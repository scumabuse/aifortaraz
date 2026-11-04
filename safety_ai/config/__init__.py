"""Configuration: pydantic models, the YAML/env/CLI loader and zones files (spec 7.2).

Public API (docs/ARCHITECTURE.md 3.2)::

    load_settings(config_path, cli_overrides=None, env=None, dotenv_path=Path(".env"))
        -> AppSettings
    load_zones_file(path, camera_id) -> CameraConfig
    format_validation_error(exc) -> list[str]
    exit_on_config_error(err: ConfigError) -> NoReturn      # prints "config error: ..." and exits 2

The models live in :mod:`safety_ai.config.models`, the loader in :mod:`safety_ai.config.loader`
(precedence CLI > env > YAML > defaults) and zones files in :mod:`safety_ai.config.zones`.
"""

from __future__ import annotations

from safety_ai.config.loader import (
    ConfigError,
    exit_on_config_error,
    format_validation_error,
    load_settings,
)
from safety_ai.config.models import AppSettings
from safety_ai.config.zones import load_zones_file

__all__ = [
    "AppSettings",
    "ConfigError",
    "exit_on_config_error",
    "format_validation_error",
    "load_settings",
    "load_zones_file",
]

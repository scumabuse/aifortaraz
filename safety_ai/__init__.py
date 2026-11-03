"""safety-ai: the AI module of the Digital Safety Inspector platform.

Watches industrial camera streams and detects occupational-safety events (PPE violations, zone
intrusion, overcrowding, vehicle proximity, falls, immobile persons) in real time, reporting them
to the website backend through the fixed integration contract in docs/CONTRACT.md.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"

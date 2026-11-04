"""Avoid persisting WebSocket query credentials in Uvicorn access logs."""

import logging
import re


class CredentialFilter(logging.Filter):
    def filter(self, record):
        def redact(value):
            return (
                re.sub(
                    r"(api\.telegram\.org/bot)[^/\s]+",
                    r"\1[REDACTED]",
                    re.sub(
                        r'([?&](?:token|api_key)=)[^&\s"\']+', r"\1[REDACTED]", value
                    ),
                )
                if isinstance(value, str)
                else value
            )

        record.msg = redact(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(redact(value) for value in record.args)
        elif isinstance(record.args, dict):
            record.args = {key: redact(value) for key, value in record.args.items()}
        return True


def configure_logging():
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error", "httpx"):
        logging.getLogger(name).addFilter(CredentialFilter())
    application = logging.getLogger("inspector")
    if not application.handlers and not logging.getLogger().handlers:
        # Uvicorn configures only its own loggers; without a handler the application's
        # INFO lines (for example the API key seeding notice) would be dropped.
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:     %(message)s"))
        handler.addFilter(CredentialFilter())
        application.addHandler(handler)
        application.setLevel(logging.INFO)

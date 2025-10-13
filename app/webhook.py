"""Utilities for delivering webhook callbacks to external services."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import random
import time
from typing import Dict

import httpx


log = logging.getLogger("app.webhook")


def _build_signature(secret: str, body: bytes) -> str:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def send_webhook(webhook_url: str, payload: Dict[str, object], secret: str | None) -> None:
    """Send the provided payload to the webhook URL with retry semantics."""

    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    if secret:
        headers["X-Signature"] = _build_signature(secret, body)

    max_attempts = 5
    base_delay = 1.0

    with httpx.Client(timeout=10.0) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                response = client.post(webhook_url, content=body, headers=headers)
            except Exception as exc:  # pragma: no cover - network failure guard
                log.warning(
                    "webhook.attempt.error",
                    extra={
                        "attempt": attempt,
                        "webhook_url": webhook_url,
                        "error": str(exc),
                    },
                )
                response = None
            else:
                if 200 <= response.status_code < 300:
                    log.info(
                        "webhook.delivered",
                        extra={
                            "attempt": attempt,
                            "status_code": response.status_code,
                            "webhook_url": webhook_url,
                        },
                    )
                    return

                log.warning(
                    "webhook.attempt.non_2xx",
                    extra={
                        "attempt": attempt,
                        "status_code": response.status_code,
                        "webhook_url": webhook_url,
                        "body": response.text[:200] if response.text else "",
                    },
                )

            if attempt >= max_attempts:
                break

            delay = base_delay * (2 ** (attempt - 1))
            jitter = random.uniform(0, delay * 0.25)
            sleep_for = delay + jitter
            log.info(
                "webhook.retry",
                extra={
                    "next_attempt": attempt + 1,
                    "sleep": round(sleep_for, 2),
                    "webhook_url": webhook_url,
                },
            )
            time.sleep(sleep_for)

    log.error(
        "webhook.failed",
        extra={"attempts": max_attempts, "webhook_url": webhook_url},
    )


"""
Alert utilities — on_failure_callback for all Hull Tactical DAGs.

Supports:
  - Airflow task log (always)
  - Slack webhook (optional — set SLACK_WEBHOOK_URL env var to enable)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def on_failure_callback(context: dict) -> None:
    """
    Generic task failure callback attached to every DAG's DEFAULT_ARGS.

    Always logs the failure to the Airflow task log.
    Optionally posts a Slack message if SLACK_WEBHOOK_URL is configured.

    Usage in DEFAULT_ARGS:
        from dags.utils.alert_utils import on_failure_callback
        DEFAULT_ARGS = {
            ...
            "on_failure_callback": on_failure_callback,
        }
    """
    ti = context["task_instance"]
    dag_id = context["dag"].dag_id
    task_id = ti.task_id
    logical_date = context.get("ds", "unknown")
    exception = context.get("exception", "unknown error")
    try_number = ti.try_number

    log_msg = (
        f"[{dag_id}] Task '{task_id}' failed "
        f"(attempt {try_number}) on {logical_date}: {exception}"
    )
    logger.error(log_msg)

    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook_url:
        return

    try:
        import requests

        slack_body = {
            "text": (
                f":red_circle: *Airflow Task Failed*\n"
                f"• DAG: `{dag_id}`\n"
                f"• Task: `{task_id}`\n"
                f"• Date: `{logical_date}`\n"
                f"• Attempt: {try_number}\n"
                f"• Error: `{str(exception)[:300]}`"
            )
        }
        resp = requests.post(webhook_url, json=slack_body, timeout=10)
        if resp.status_code != 200:
            logger.warning(
                "Slack alert returned HTTP %s: %s", resp.status_code, resp.text[:200]
            )
    except Exception as exc:
        logger.warning("Could not send Slack alert: %s", exc)

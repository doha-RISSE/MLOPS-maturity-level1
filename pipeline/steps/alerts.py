"""Schema-skew alerting."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__version__ = "1.0.0"
LOGGER = logging.getLogger(__name__)


def create_schema_alert(execution_run_id: str, input_path: str, issues: list[dict[str, Any]], alert_dir: str | Path = "/app/alerts") -> Path:
    """Persist and visibly announce a schema validation failure."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    alert = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "STOPPED_SCHEMA_SKEW", "execution_run_id": execution_run_id, "input_path": input_path, "issues": issues, "message": "SCHEMA SKEW DETECTED : data scientist intervention required"}
    directory = Path(alert_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"alert_{timestamp}_{execution_run_id}.json"
    output.write_text(json.dumps(alert, indent=2), encoding="utf-8")
    lines = ["=" * 72, "SCHEMA SKEW DETECTED : data scientist intervention required", f"Input: {input_path}"]
    lines.extend(f"{issue['type']} | {issue['columns']} | {issue['details']}" for issue in issues)
    lines.append("=" * 72)
    message = "\n".join(lines)
    LOGGER.error(message)
    print(message)
    return output

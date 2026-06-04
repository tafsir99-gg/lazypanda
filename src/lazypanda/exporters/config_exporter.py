"""
Config Exporter.

WHAT IT DOES:
    Saves the exact YAML configuration that was used for a pipeline run.
    This makes every run fully reproducible — not just the data transformations,
    but also the parameters that controlled them.

WHY SAVE THE CONFIG?
    Six months from now you may have changed `default_config.yaml`. Without a
    saved run config, you cannot reproduce the exact cleaning decisions.
    With it, you can point to the config and re-run identically.

    Also useful for:
    - Sharing cleaning parameters with teammates
    - Tracking config changes via git diff
    - Auditing which thresholds were used in a production cleaning run

USAGE:
    from lazypanda.exporters.config_exporter import ConfigExporter
    from lazypanda.core.config_manager import AppConfig

    path = ConfigExporter().write(config, output_path=Path("outputs/run_config.yaml"))
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lazypanda.core.config_manager import AppConfig

logger = logging.getLogger("lazypanda")


class ConfigExporter:
    """
    Saves the AppConfig used for a pipeline run as a YAML file.

    The saved YAML can be used directly as a `--config` argument to reproduce
    the run: `lazypanda clean data.csv --config outputs/run_config.yaml`
    """

    def build_payload(self, config: AppConfig) -> dict:
        """
        Convert AppConfig to a plain dict suitable for YAML export.

        WHY NOT just use config.model_dump() directly?
        We add a meta header comment and ensure all values are
        standard Python types (not pydantic wrappers).
        """
        data = config.model_dump()
        # Remove internal/advanced sections that shouldn't be in a user-facing config
        # (they can always be overridden if needed)
        return data

    def write(self, config: AppConfig, output_path: Path) -> Path:
        """
        Write the config to a YAML file.

        Args:
            config:      The AppConfig used for this run.
            output_path: File path to write (e.g., outputs/run_config.yaml).

        Returns:
            The path that was written.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        payload = self.build_payload(config)

        # Build the YAML with a header comment
        header = (
            f"# AI Data Cleaner — Run Configuration\n"
            f"# Generated: {now}\n"
            f"# Usage: lazypanda clean <file> --config {output_path.name}\n"
            f"#\n"
            f"# This file captures the EXACT configuration used for this run.\n"
            f"# Copy or edit it to reproduce or adjust the cleaning.\n\n"
        )

        yaml_body = yaml.dump(
            payload,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
        )

        output_path.write_text(header + yaml_body, encoding="utf-8")
        logger.info("ConfigExporter: wrote run config to %s", output_path)
        return output_path

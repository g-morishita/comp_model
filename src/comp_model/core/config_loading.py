"""Load declarative configuration files for public CLIs and APIs.

The loader supports JSON and YAML mappings with strict root-type validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUPPORTED_CONFIG_SUFFIXES: tuple[str, ...] = (".json", ".yaml", ".yml")


def load_config_mapping(path: str | Path) -> dict[str, Any]:
    """Load one JSON/YAML config file whose root is an object mapping.

    Parameters
    ----------
    path : str | pathlib.Path
        Config file path. Supported suffixes are `.json`, `.yaml`, and `.yml`.

    Returns
    -------
    dict[str, Any]
        Parsed config mapping.

    Raises
    ------
    ValueError
        If suffix is unsupported or config root is not an object mapping.
    ImportError
        If YAML parsing is requested without PyYAML installed.
    """

    config_path = Path(path)
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised only without pyyaml
            raise ImportError(
                "YAML config loading requires PyYAML. Install with `pip install pyyaml`."
            ) from exc
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    else:
        supported = ", ".join(SUPPORTED_CONFIG_SUFFIXES)
        raise ValueError(
            f"unsupported config file extension {suffix!r}; expected one of {supported}"
        )

    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON/YAML object")
    return raw


__all__ = ["SUPPORTED_CONFIG_SUFFIXES", "load_config_mapping"]

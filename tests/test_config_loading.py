"""Tests for JSON/YAML config loading helpers."""

from __future__ import annotations

import json

import pytest

from comp_model.core import load_config_mapping


def test_load_config_mapping_accepts_json(tmp_path) -> None:
    """Loader should parse JSON config objects."""

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"a": 1, "b": {"c": 2}}), encoding="utf-8")

    loaded = load_config_mapping(path)
    assert loaded == {"a": 1, "b": {"c": 2}}


def test_load_config_mapping_accepts_yaml(tmp_path) -> None:
    """Loader should parse YAML config objects."""

    path = tmp_path / "config.yaml"
    path.write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")

    loaded = load_config_mapping(path)
    assert loaded == {"a": 1, "b": {"c": 2}}


def test_load_config_mapping_rejects_unsupported_extension(tmp_path) -> None:
    """Loader should fail fast on unknown config file suffix."""

    path = tmp_path / "config.toml"
    path.write_text("a = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported config file extension"):
        load_config_mapping(path)


def test_load_config_mapping_requires_mapping_root(tmp_path) -> None:
    """Loader should reject non-mapping top-level config payloads."""

    path = tmp_path / "config.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    with pytest.raises(ValueError, match="config root must be a JSON/YAML object"):
        load_config_mapping(path)

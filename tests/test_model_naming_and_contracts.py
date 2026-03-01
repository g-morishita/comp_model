"""Tests for canonical model naming and contract docstrings."""

from __future__ import annotations

import inspect

import pytest

import comp_model.models as model_module
from comp_model.plugins import build_default_registry


def test_deprecated_component_ids_removed() -> None:
    """Deprecated model IDs should no longer resolve in the plugin registry."""

    registry = build_default_registry()

    with pytest.raises(KeyError):
        registry.create_model("q_learning")

    with pytest.raises(KeyError):
        registry.create_model("random_agent")

    with pytest.raises(KeyError):
        registry.create_model("qrl")

    with pytest.raises(KeyError):
        registry.create_model("qrl_stay")

    with pytest.raises(KeyError):
        registry.create_model("unidentifiable_qrl")


def test_canonical_model_docstrings_share_contract_sections() -> None:
    """Every exported *Model class should expose the same contract sections."""

    required_sections = ("Model Contract", "Decision Rule", "Update Rule")

    exported_models: list[type] = []
    for name in getattr(model_module, "__all__", []):
        attribute = getattr(model_module, name)
        if inspect.isclass(attribute) and name.endswith("Model"):
            exported_models.append(attribute)

    assert exported_models, "No exported model classes found"

    for model_cls in exported_models:
        docstring = model_cls.__doc__ or ""
        for section in required_sections:
            assert section in docstring, f"{model_cls.__name__} missing section: {section}"

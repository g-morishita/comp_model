"""Tests for canonical model naming and contract docstrings."""

from __future__ import annotations

import pytest

from comp_model.models import (
    AsocialQValueSoftmaxModel,
    AsocialStateQValueSoftmaxModel,
    AsocialStateQValueSoftmaxPerseverationModel,
    AsocialStateQValueSoftmaxSplitAlphaModel,
    SocialSelfOutcomeValueShapingModel,
    UniformRandomPolicyModel,
)
from comp_model.plugins import build_default_registry


def test_legacy_component_ids_removed() -> None:
    """Legacy model IDs should no longer resolve in the plugin registry."""

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
    """Canonical model docs should expose a consistent contract template."""

    canonical_models = (
        UniformRandomPolicyModel,
        AsocialQValueSoftmaxModel,
        AsocialStateQValueSoftmaxModel,
        AsocialStateQValueSoftmaxPerseverationModel,
        AsocialStateQValueSoftmaxSplitAlphaModel,
        SocialSelfOutcomeValueShapingModel,
    )

    required_sections = ("Model Contract", "Decision Rule", "Update Rule")
    for model_cls in canonical_models:
        docstring = model_cls.__doc__ or ""
        for section in required_sections:
            assert section in docstring, f"{model_cls.__name__} missing section: {section}"

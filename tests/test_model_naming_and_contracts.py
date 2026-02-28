"""Tests for canonical model naming and contract docstrings."""

from __future__ import annotations

import pytest

from comp_model.models import (
    AsocialQValueSoftmaxModel,
    AsocialStateQValueSoftmaxModel,
    AsocialStateQValueSoftmaxPerseverationModel,
    AsocialStateQValueSoftmaxSplitAlphaModel,
    QLearningAgent,
    QRL,
    QRL_Stay,
    RandomAgent,
    UnidentifiableQRL,
    UniformRandomPolicyModel,
)


def test_deprecated_class_aliases_emit_warning_and_still_work() -> None:
    """Legacy class names should remain functional with deprecation warnings."""

    with pytest.warns(DeprecationWarning, match="QLearningAgent"):
        legacy_q = QLearningAgent()

    with pytest.warns(DeprecationWarning, match="RandomAgent"):
        legacy_random = RandomAgent()

    with pytest.warns(DeprecationWarning, match="QRL"):
        legacy_qrl = QRL()

    with pytest.warns(DeprecationWarning, match="QRL_Stay"):
        legacy_qrl_stay = QRL_Stay()

    with pytest.warns(DeprecationWarning, match="UnidentifiableQRL"):
        legacy_split = UnidentifiableQRL()

    assert isinstance(legacy_q, AsocialQValueSoftmaxModel)
    assert isinstance(legacy_random, UniformRandomPolicyModel)
    assert isinstance(legacy_qrl, AsocialStateQValueSoftmaxModel)
    assert isinstance(legacy_qrl_stay, AsocialStateQValueSoftmaxPerseverationModel)
    assert isinstance(legacy_split, AsocialStateQValueSoftmaxSplitAlphaModel)


def test_canonical_model_docstrings_share_contract_sections() -> None:
    """Canonical model docs should expose a consistent contract template."""

    canonical_models = (
        UniformRandomPolicyModel,
        AsocialQValueSoftmaxModel,
        AsocialStateQValueSoftmaxModel,
        AsocialStateQValueSoftmaxPerseverationModel,
        AsocialStateQValueSoftmaxSplitAlphaModel,
    )

    required_sections = ("Model Contract", "Decision Rule", "Update Rule")
    for model_cls in canonical_models:
        docstring = model_cls.__doc__ or ""
        for section in required_sections:
            assert section in docstring, f"{model_cls.__name__} missing section: {section}"

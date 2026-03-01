"""Tests for explicit source-to-canonical model parity mapping."""

from __future__ import annotations

import comp_model.models as models_pkg

from comp_model.models import MODEL_PARITY
from comp_model.plugins import build_default_registry


def test_parity_entries_cover_expected_source_model_names() -> None:
    """Parity table should include every targeted source model name exactly once."""

    expected_source_names = {
        "QRL",
        "QRL_Stay",
        "UnidentifiableQRL",
        "VS",
        "Vicarious_RL",
        "Vicarious_RL_Stay",
        "Vicarious_VS",
        "Vicarious_VS_Stay",
        "AP_RL_NoStay",
        "AP_RL_Stay",
        "Vicarious_AP_VS",
        "Vicarious_AP_DB_STAY",
        "Vicarious_Dir_DB_Stay",
        "Vicarious_DB_Stay",
        "VicQ_AP_DualW_NoStay",
        "VicQ_AP_DualW_Stay",
        "VicQ_AP_IndepDualW",
        "ConditionedSharedDeltaModel",
        "ConditionedSharedDeltaSocialModel",
    }
    observed_source_names = {entry.source_name for entry in MODEL_PARITY}

    assert observed_source_names == expected_source_names
    assert len(observed_source_names) == len(MODEL_PARITY)


def test_implemented_parity_entries_resolve_to_classes_and_optional_plugin_ids() -> None:
    """Implemented mappings should resolve to classes and plugin IDs when provided."""

    registry = build_default_registry()
    available_model_ids = {manifest.component_id for manifest in registry.list("model")}

    for entry in MODEL_PARITY:
        if entry.status != "implemented":
            continue

        assert entry.canonical_class_name is not None
        assert hasattr(models_pkg, entry.canonical_class_name)

        if entry.canonical_component_id is not None:
            assert entry.canonical_component_id in available_model_ids


def test_no_planned_entries_remain_in_model_parity_matrix() -> None:
    """All declared model families should now be mapped as implemented."""

    assert [entry for entry in MODEL_PARITY if entry.status == "planned"] == []

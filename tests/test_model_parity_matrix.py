"""Tests for explicit v1-to-canonical model parity mapping."""

from __future__ import annotations

import comp_model.models as models_pkg

from comp_model.models import V1_MODEL_PARITY
from comp_model.plugins import build_default_registry


def test_v1_parity_entries_cover_expected_legacy_model_names() -> None:
    """Parity table should include every targeted v1 model name exactly once."""

    expected_legacy_names = {
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
    observed_legacy_names = {entry.legacy_name for entry in V1_MODEL_PARITY}

    assert observed_legacy_names == expected_legacy_names
    assert len(observed_legacy_names) == len(V1_MODEL_PARITY)


def test_implemented_parity_entries_resolve_to_registered_model_ids_and_classes() -> None:
    """Implemented mappings should point to existing model classes and plugin IDs."""

    registry = build_default_registry()
    available_model_ids = {manifest.component_id for manifest in registry.list("model")}

    for entry in V1_MODEL_PARITY:
        if entry.status != "implemented":
            continue

        assert entry.canonical_component_id is not None
        assert entry.canonical_class_name is not None
        assert entry.canonical_component_id in available_model_ids
        assert hasattr(models_pkg, entry.canonical_class_name)


def test_planned_parity_entries_are_explicitly_marked_unimplemented() -> None:
    """Planned mappings should not expose canonical IDs/classes yet."""

    planned = [entry for entry in V1_MODEL_PARITY if entry.status == "planned"]

    assert {entry.legacy_name for entry in planned} == {
        "ConditionedSharedDeltaModel",
        "ConditionedSharedDeltaSocialModel",
    }
    for entry in planned:
        assert entry.canonical_component_id is None
        assert entry.canonical_class_name is None

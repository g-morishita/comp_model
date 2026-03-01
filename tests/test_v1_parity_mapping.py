"""Tests for explicit v1 parity mapping artifacts."""

from __future__ import annotations

from comp_model.parity import V1_MODEL_PARITY_MAP
from comp_model.plugins import build_default_registry


def test_v1_parity_map_covers_all_v1_public_model_symbols() -> None:
    """Parity map should include every v1 public model/wrapper symbol."""

    expected = {
        "QRL",
        "QRL_Stay",
        "UnidentifiableQRL",
        "VS",
        "Vicarious_VS",
        "Vicarious_VS_Stay",
        "Vicarious_RL",
        "Vicarious_RL_Stay",
        "AP_RL_Stay",
        "AP_RL_NoStay",
        "Vicarious_AP_VS",
        "VicQ_AP_DualW_Stay",
        "VicQ_AP_DualW_NoStay",
        "VicQ_AP_IndepDualW",
        "Vicarious_AP_DB_STAY",
        "Vicarious_Dir_DB_Stay",
        "Vicarious_DB_Stay",
        "ConditionedSharedDeltaModel",
        "ConditionedSharedDeltaSocialModel",
        "wrap_model_with_shared_delta_conditions",
    }
    observed = {entry.legacy_name for entry in V1_MODEL_PARITY_MAP}
    assert observed == expected


def test_v1_parity_map_has_unique_legacy_names() -> None:
    """Legacy model names in parity map should be unique."""

    names = [entry.legacy_name for entry in V1_MODEL_PARITY_MAP]
    assert len(names) == len(set(names))


def test_v1_component_mappings_resolve_in_registry() -> None:
    """Mapped v1 entries should resolve to registered v2 model components."""

    registry = build_default_registry()
    registered = {manifest.component_id for manifest in registry.list("model")}

    mapped_ids = {
        entry.replacement_component_id
        for entry in V1_MODEL_PARITY_MAP
        if entry.replacement_component_id is not None
    }
    assert mapped_ids.issubset(registered)


def test_v1_component_mappings_can_instantiate_defaults() -> None:
    """Every mapped v2 component should instantiate with default args."""

    registry = build_default_registry()
    for entry in V1_MODEL_PARITY_MAP:
        if entry.replacement_component_id is None:
            continue
        instance = registry.create_model(entry.replacement_component_id)
        assert instance is not None

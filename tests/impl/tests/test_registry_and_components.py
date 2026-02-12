import ast
from pathlib import Path

import numpy as np
import pytest

from comp_model_impl.register import make_registry
from comp_model_impl.bandits.bernoulli import BernoulliBanditEnv
from comp_model_impl.demonstrators.fixed_sequence import FixedSequenceDemonstrator
from comp_model_impl.demonstrators.noisy_best import NoisyBestArmDemonstrator
from comp_model_impl.demonstrators.rl_agent import RLDemonstrator
from comp_model_impl.models import (
    QRL,
    VS,
    VicQ_AP_DualW_Stay,
    VicQ_AP_DualW_NoStay,
    VicQ_AP_IndepDualW,
    Vicarious_RL,
    Vicarious_VS,
    Vicarious_VS_Stay,
    UnidentifiableQRL,
)


def test_make_registry_contains_expected_components():
    r = make_registry()

    # Models
    assert "QRL" in r.models.names()
    assert r.models.get("QRL") is QRL

    assert "VS" in r.models.names()
    assert r.models.get("VS") is VS

    assert "Vicarious_RL" in r.models.names()
    assert r.models.get("Vicarious_RL") is Vicarious_RL

    assert "Vicarious_VS" in r.models.names()
    assert r.models.get("Vicarious_VS") is Vicarious_VS

    assert "Vicarious_VS_Stay" in r.models.names()
    assert r.models.get("Vicarious_VS_Stay") is Vicarious_VS_Stay

    assert "VicQ_AP_DualW_Stay" in r.models.names()
    assert r.models.get("VicQ_AP_DualW_Stay") is VicQ_AP_DualW_Stay
    assert "VicQ_AP_DualW_NoStay" in r.models.names()
    assert r.models.get("VicQ_AP_DualW_NoStay") is VicQ_AP_DualW_NoStay
    assert "VicQ_AP_IndepDualW" in r.models.names()
    assert r.models.get("VicQ_AP_IndepDualW") is VicQ_AP_IndepDualW

    # Bandits
    assert "BernoulliBanditEnv" in r.bandits.names()
    assert r.bandits.get("BernoulliBanditEnv") is BernoulliBanditEnv

    # Demonstrators
    assert "FixedSequenceDemonstrator" in r.demonstrators.names()
    assert r.demonstrators.get("FixedSequenceDemonstrator") is FixedSequenceDemonstrator
    assert "NoisyBestArmDemonstrator" in r.demonstrators.names()
    assert r.demonstrators.get("NoisyBestArmDemonstrator") is NoisyBestArmDemonstrator
    assert "RLDemonstrator" in r.demonstrators.names()
    assert r.demonstrators.get("RLDemonstrator") is RLDemonstrator


def test_registry_covers_all_exported_models():
    r = make_registry()
    expected = _discover_model_class_names()
    missing = sorted(name for name in expected if name not in r.models.names())
    assert not missing, f"Registry missing models: {missing}"


def test_registry_covers_all_exported_bandits():
    r = make_registry()
    expected = _discover_bandit_class_names()
    missing = sorted(name for name in expected if name not in r.bandits.names())
    assert not missing, f"Registry missing bandits: {missing}"


def test_registry_covers_all_exported_demonstrators():
    r = make_registry()
    expected = _discover_demonstrator_class_names()
    missing = sorted(name for name in expected if name not in r.demonstrators.names())
    assert not missing, f"Registry missing demonstrators: {missing}"


def _discover_model_class_names() -> set[str]:
    return _discover_class_names(
        subdir="models",
        base_names={"ComputationalModel"},
        skip={"ConditionedSharedDeltaModel", "ConditionedSharedDeltaSocialModel"},
    )


def _discover_bandit_class_names() -> set[str]:
    return _discover_class_names(subdir="bandits", base_names={"BanditEnv"})


def _discover_demonstrator_class_names() -> set[str]:
    return _discover_class_names(subdir="demonstrators", base_names={"Demonstrator"}, skip={"FixedSequenceDemonstrator"})


def _discover_class_names(*, subdir: str, base_names: set[str], skip: set[str] | None = None) -> set[str]:
    root = Path(__file__).resolve().parents[3]
    target_dir = root / "comp_model_impl" / "src" / "comp_model_impl" / subdir
    if not target_dir.exists():
        raise AssertionError(f"Directory not found: {target_dir}")

    skip = skip or set()
    names: set[str] = set()
    for path in target_dir.rglob("*.py"):
        if path.name.startswith("_"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name.startswith("_") or node.name in skip:
                continue
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id in base_names:
                    names.add(node.name)
                    break
                if isinstance(base, ast.Attribute) and base.attr in base_names:
                    names.add(node.name)
                    break
    return names


def test_bernoulli_bandit_env_validation_and_step():
    with pytest.raises(ValueError):
        BernoulliBanditEnv(probs=[0.5])

    with pytest.raises(ValueError):
        BernoulliBanditEnv(probs=[-0.1, 0.2])

    env = BernoulliBanditEnv(probs=[0.0, 1.0, 0.5])
    spec = env.spec
    assert spec.n_actions == 3
    assert spec.outcome_range == (0.0, 1.0)
    assert spec.outcome_is_bounded is True
    assert spec.n_states == 1

    rng = np.random.default_rng(123)
    env.reset(rng=rng)
    assert env.get_state() == 0

    # Arm 0 always yields 0, arm 1 always yields 1.
    assert env.step(action=0, rng=rng).outcome == 0.0
    assert env.step(action=1, rng=rng).outcome == 1.0

    # Arm 2 is stochastic but bounded.
    out = [env.step(action=2, rng=rng).outcome for _ in range(50)]
    assert set(out).issubset({0.0, 1.0})


def test_bernoulli_bandit_env_from_config():
    cfg = {"probs": [0.1, 0.9]}
    env = BernoulliBanditEnv.from_config(cfg)
    assert isinstance(env, BernoulliBanditEnv)
    assert list(env.probs) == [0.1, 0.9]

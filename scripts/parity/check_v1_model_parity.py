#!/usr/bin/env python3
"""Validate v1-to-v2 model parity mapping against local repositories.

Usage
-----
python scripts/parity/check_v1_model_parity.py --v1-root /path/to/comp_model_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from comp_model.parity import V1_MODEL_PARITY_MAP
from comp_model.plugins import build_default_registry


def _load_v1_models_module(v1_root: Path):
    """Import `comp_model_impl.models` from a local v1 checkout."""

    impl_src = v1_root / "comp_model_impl" / "src"
    core_src = v1_root / "comp_model_core" / "src"

    if not impl_src.exists():
        raise FileNotFoundError(f"v1 path missing: {impl_src}")
    if not core_src.exists():
        raise FileNotFoundError(f"v1 path missing: {core_src}")

    sys.path.insert(0, str(impl_src))
    sys.path.insert(0, str(core_src))
    import comp_model_impl.models as v1_models  # type: ignore

    return v1_models


def main() -> int:
    """Run parity validation and print a compact report."""

    parser = argparse.ArgumentParser(description="Check v1->v2 model parity mapping.")
    parser.add_argument(
        "--v1-root",
        default="/Users/morishitag/comp_model_v1",
        help="Path to local v1 repository root.",
    )
    args = parser.parse_args()

    v1_root = Path(args.v1_root).expanduser().resolve()
    v1_models = _load_v1_models_module(v1_root)
    registry = build_default_registry()
    registered_model_ids = {manifest.component_id for manifest in registry.list("model")}

    missing_v1_symbols: list[str] = []
    missing_v2_components: list[str] = []

    for entry in V1_MODEL_PARITY_MAP:
        if not hasattr(v1_models, entry.legacy_name):
            missing_v1_symbols.append(entry.legacy_name)
        if entry.replacement_component_id is not None and entry.replacement_component_id not in registered_model_ids:
            missing_v2_components.append(entry.replacement_component_id)

    print(f"v1 root: {v1_root}")
    print(f"mapping rows: {len(V1_MODEL_PARITY_MAP)}")
    print(f"mapped v2 model components: {len({e.replacement_component_id for e in V1_MODEL_PARITY_MAP if e.replacement_component_id is not None})}")

    if missing_v1_symbols:
        print("Missing v1 symbols:")
        for item in missing_v1_symbols:
            print(f"  - {item}")

    if missing_v2_components:
        print("Missing v2 components:")
        for item in missing_v2_components:
            print(f"  - {item}")

    if missing_v1_symbols or missing_v2_components:
        return 1

    print("Parity mapping check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

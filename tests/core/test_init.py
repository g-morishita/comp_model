import types

import pytest


def test_lazy_getattr_imports_submodule():
    import comp_model_core as cmc

    mod = getattr(cmc, "params")
    assert isinstance(mod, types.ModuleType)
    assert mod.__name__ == "comp_model_core.params"


def test_lazy_getattr_unknown_raises():
    import comp_model_core as cmc

    with pytest.raises(AttributeError, match="no attribute"):
        getattr(cmc, "does_not_exist")


def test_dir_includes_all_and_is_sorted():
    import comp_model_core as cmc

    names = dir(cmc)
    for name in cmc.__all__:
        assert name in names
    assert names == sorted(names)

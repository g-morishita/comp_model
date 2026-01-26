def test_can_import_top_level():
    import comp_model_core  # noqa: F401


def test_can_import_submodules():
    import comp_model_core.data  # noqa: F401
    import comp_model_core.interfaces  # noqa: F401
    import comp_model_core.plans  # noqa: F401
    import comp_model_core.params  # noqa: F401

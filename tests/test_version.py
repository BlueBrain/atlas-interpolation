import atlinter


def test_version_exists():
    # Version exists
    assert hasattr(atlinter, "__version__")
    assert isinstance(atlinter.__version__, str)
    parts = atlinter.__version__.split(".")

    # Version has correct format
    # allow for an optional ".devXX" part for local testing
    assert len(parts) in {3, 4}
    assert parts[0].isdecimal()  # major
    assert parts[1].isdecimal()  # minor
    assert parts[2].isdecimal()  # patch

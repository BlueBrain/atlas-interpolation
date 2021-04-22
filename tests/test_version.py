import atlinter


def test_version_exists():
    # Version exists
    assert hasattr(atlinter, "__version__")
    assert isinstance(atlinter.__version__, str)
    parts = atlinter.__version__.split(".")

    # Version has correct format
    assert len(parts) == 3
    assert parts[0].isdecimal()  # major
    assert parts[1].isdecimal()  # minor

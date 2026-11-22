"""Bootstrap test ensuring the package can be imported."""

from mlsecops import __version__


def test_package_version() -> None:
    """The package exposes an initial, fixed semantic version."""
    assert __version__ == "0.1.0"

"""Check committed DVC metadata and optional data recovery without secrets.

The metadata mode runs in CI without a DVC remote or downloaded datasets.
The data and shared-remote modes are intended for local verification.
"""

from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, cast
from urllib.parse import urlsplit


CheckMode = Literal["metadata", "data", "shared-remote"]

# Files must be present before starting the Docker Compose application.
REQUIRED_DATA_FILES = (
    "data/raw/mlsecops_prep_set.csv",
    "data/processed/train.csv",
    "data/processed/test.csv",
    "data/validation/fit.csv",
    "data/validation/validation.csv",
    "data/validation/validation_protocol.json",
)

# Filesystem and file:// remotes cannot be pulled from an arbitrary fresh clone.
SHARED_SCHEMES = frozenset({"s3", "gs", "azure", "ssh", "gdrive", "webdav", "webdavs", "oss"})


@dataclass(frozen=True)
class VerificationResult:
    """A deterministic result suitable for CLI output and unit tests."""

    errors: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def shared_remote_configured(config_path: Path) -> bool:
    """Return whether committed DVC config declares a shareable default remote."""
    if not config_path.is_file():
        return False

    parser = configparser.RawConfigParser()
    try:
        with config_path.open(encoding="utf-8") as config_file:
            parser.read_file(config_file)
    except (OSError, configparser.Error):
        return False

    default_remote = parser.get("core", "remote", fallback="").strip()
    if not default_remote:
        return False

    section = f'remote "{default_remote}"'
    url = parser.get(section, "url", fallback="").strip()
    return urlsplit(url).scheme.lower() in SHARED_SCHEMES


def verify_repository(root: Path, mode: CheckMode = "metadata") -> VerificationResult:
    """Validate files needed for each reproducibility level.

    Metadata-only checks never download or read the original private dataset.
    """
    errors: list[str] = []
    notes: list[str] = []

    for filename in ("pyproject.toml", "uv.lock", "compose.yaml", ".dvc/config"):
        if not (root / filename).is_file():
            errors.append(f"Missing required repository file: {filename}")

    version_file = root / ".python-version"
    if not version_file.is_file() or not version_file.read_text(encoding="utf-8").strip().startswith("3.13"):
        errors.append(".python-version must pin Python 3.13")

    data_dir = root / "data"
    pointers = sorted(data_dir.rglob("*.dvc")) if data_dir.is_dir() else []
    if (root / "data.dvc").is_file():
        pointers.append(root / "data.dvc")
    if not pointers:
        errors.append("No DVC pointer found under data/. Run `dvc add` and commit its .dvc files.")

    # A dataset folder can also be tracked as a single DVC directory pointer.
    raw_pointer_paths = (
        root / "data/raw/mlsecops_prep_set.csv.dvc",
        root / "data/raw.dvc",
        root / "data.dvc",
    )
    if not any(path.is_file() for path in raw_pointer_paths):
        errors.append("Raw dataset is not tracked by a recognized DVC pointer.")

    if mode == "metadata":
        notes.append("Metadata verified; downloaded data and remote accessibility were not checked.")
    elif mode == "data":
        for filename in REQUIRED_DATA_FILES:
            if not (root / filename).is_file():
                errors.append(f"Missing dataset file: {filename}; try `uv run --locked dvc pull`.")
        notes.append("Local data presence verified; a fresh-clone pull still requires remote access.")
    elif mode == "shared-remote":
        if not shared_remote_configured(root / ".dvc/config"):
            errors.append(
                "No shared default DVC remote in committed .dvc/config. "
                "A remote in .dvc/config.local is only available on this machine."
            )
        notes.append("Remote configuration checked, but credentials and actual dvc pull were not tested.")

    return VerificationResult(tuple(errors), tuple(notes))


def main(argv: Sequence[str] | None = None) -> int:
    """Run a reproducibility preflight; return a shell-friendly exit code."""
    parser = argparse.ArgumentParser(description="Validate MLSecOps reproducibility prerequisites")
    parser.add_argument("--mode", choices=("metadata", "data", "shared-remote"), default="metadata")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    result = verify_repository(arguments.root, cast(CheckMode, arguments.mode))

    for note in result.notes:
        print(f"INFO: {note}")
    for error in result.errors:
        print(f"ERROR: {error}")
    print("PASS: reproducibility preflight" if result.ok else "FAIL: reproducibility preflight")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Small representative dataset shared by fast, isolated unit tests."""

import pandas as pd
import pytest


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Return raw-format rows without depending on DVC data in CI."""
    return pd.DataFrame(
        {
            "language": [" Python ", "TypeScript", "Go", "Python"] * 8,
            "stars": list(range(100, 132)),
            "forks": list(range(10, 42)),
            "open_issues": [0, 3, 10, 15] * 8,
            "age_days": list(range(200, 232)),
            "has_coc": [True, False, "true", "0"] * 8,
            "has_contributing": ["False", "True", 0, 1] * 8,
            "has_workflows": [1, 0, True, False] * 8,
            "has_readme": ["true", "false", "1", "0"] * 8,
        }
    )

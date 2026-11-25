# Run from the repository root in Windows PowerShell after configuring a shared DVC remote.
# The checks fail safely: they do not remove existing data or Docker volumes.
$ErrorActionPreference = 'Stop'

uv run --locked python -m src.reproducibility --mode metadata
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run --locked python -m src.reproducibility --mode shared-remote
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run --locked dvc pull
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

uv run --locked python -m src.reproducibility --mode data
exit $LASTEXITCODE

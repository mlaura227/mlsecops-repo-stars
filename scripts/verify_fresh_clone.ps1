# Isolated fresh-clone smoke test: no project data or Git state is modified.
# Run after pushing/merging the committed shared DVC remote configuration.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$RepoUrl,
    [ValidateNotNullOrEmpty()]
    [string]$Branch = 'main',
    [switch]$KeepClone
)

$ErrorActionPreference = 'Stop'

# S3-compatible storage (including DagsHub) needs credentials at fetch time.
# Never pass access keys as command-line arguments or commit them to the repo.
if (-not $env:AWS_ACCESS_KEY_ID -or -not $env:AWS_SECRET_ACCESS_KEY) {
    throw 'Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your local environment for this verification.'
}

$clonePath = Join-Path ([System.IO.Path]::GetTempPath()) (
    'mlsecops-dvc-smoke-' + [guid]::NewGuid().ToString('N')
)
$success = $false

try {
    & git clone --quiet --single-branch --branch $Branch $RepoUrl $clonePath
    if ($LASTEXITCODE -ne 0) { throw 'Git clone failed.' }

    Push-Location $clonePath
    try {
        & uv sync --locked
        if ($LASTEXITCODE -ne 0) { throw 'Locked uv installation failed.' }

        & uv run --locked python -m src.reproducibility --mode shared-remote
        if ($LASTEXITCODE -ne 0) { throw 'Shared DVC remote preflight failed.' }

        & uv run --locked dvc pull
        if ($LASTEXITCODE -ne 0) { throw 'DVC pull failed.' }

        & uv run --locked python -m src.reproducibility --mode data
        if ($LASTEXITCODE -ne 0) { throw 'Downloaded dataset verification failed.' }
    }
    finally {
        Pop-Location
    }

    $success = $true
    Write-Host 'PASS: fresh clone, locked dependencies, DVC pull and datasets verified.'
}
finally {
    if ($success -and -not $KeepClone) {
        Remove-Item -LiteralPath $clonePath -Recurse -Force
    }
    elseif (Test-Path -LiteralPath $clonePath) {
        Write-Warning "Clone retained for inspection: $clonePath"
    }
}

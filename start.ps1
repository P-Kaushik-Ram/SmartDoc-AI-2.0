# Local demonstration launcher. Does not install packages or alter the database.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
# This launcher deliberately uses --noreload for predictable local model memory.
# Refuse an occupied port: an older process may still have cached Python/templates
# even though the source and static files on disk have changed.
$smartdocListeners = @(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)
if ($smartdocListeners.Count -gt 0) {
    $smartdocListenerIds = ($smartdocListeners | Select-Object -ExpandProperty OwningProcess -Unique) -join ', '
    throw "Port 8000 is already served by PID(s) $smartdocListenerIds. Stop the existing server (Ctrl+C in its terminal), then run .\start.ps1 again. Python and template changes require a server restart."
}
if (-not $env:SMARTDOC_HANDWRITING_MODEL_PATH) {
    foreach ($smartdocCheckpoint in @('trocr-base-handwritten', 'trocr-small-handwritten')) {
        $smartdocModelPath = Join-Path $env:LOCALAPPDATA "SmartDocAI\models\$smartdocCheckpoint"
        if ((Test-Path -LiteralPath (Join-Path $smartdocModelPath 'preprocessor_config.json')) -and
            ((Test-Path -LiteralPath (Join-Path $smartdocModelPath 'model.safetensors')) -or
             (Test-Path -LiteralPath (Join-Path $smartdocModelPath 'pytorch_model.bin')))) {
            $env:SMARTDOC_HANDWRITING_MODEL_PATH = $smartdocModelPath
            break
        }
    }
}
python manage.py check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python manage.py runserver 127.0.0.1:8000 --noreload

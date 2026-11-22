# Use CPython's own venv selection mechanism, also in Windows sessions where the
# nested venv redirector cannot initialize native DLLs. No system packages are added.
$venvConfig = Join-Path $PSScriptRoot '.venv/pyvenv.cfg'
$baseExe = (Select-String -LiteralPath $venvConfig -Pattern '^executable = (.+)$').Matches[0].Groups[1].Value
$projectPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
$originalLauncher = $env:__PYVENV_LAUNCHER__
try {
    $env:__PYVENV_LAUNCHER__ = $projectPython
    & $baseExe @args
    $result = $LASTEXITCODE
} finally {
    $env:__PYVENV_LAUNCHER__ = $originalLauncher
}
exit $result

# Windows 只读学习/数据演练入口；真实训练在内网 Linux。
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $projectRoot 'src'
$env:PYTHONUTF8 = '1'
Push-Location $projectRoot
try {
    & python -m ticket_sft.intranet @args
    $resultCode = $LASTEXITCODE
} finally { Pop-Location }
exit $resultCode

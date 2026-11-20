# flash.ps1 -- one-click build / flash / monitor for DialectVoice_SZPI
# NOTE: keep this file pure ASCII (PowerShell 5.1 mis-parses non-ASCII comments).
#
# Usage (run from anywhere):
#   powershell -ExecutionPolicy Bypass -File .\flash.ps1                  # auto-detect port, build+flash
#   powershell -ExecutionPolicy Bypass -File .\flash.ps1 -Port COM7       # explicit port
#   powershell -ExecutionPolicy Bypass -File .\flash.ps1 -Port COM7 -Monitor
#   powershell -ExecutionPolicy Bypass -File .\flash.ps1 -Erase -Port COM7 -Monitor
#   powershell -ExecutionPolicy Bypass -File .\flash.ps1 -Clean           # fullclean first
#   powershell -ExecutionPolicy Bypass -File .\flash.ps1 -Menuconfig
#   powershell -ExecutionPolicy Bypass -File .\flash.ps1 -Reconfig        # delete sdkconfig* then rebuild

param(
    [string]$Port = "",
    [switch]$Monitor,
    [switch]$Clean,
    [switch]$Erase,
    [switch]$Menuconfig,
    [switch]$Reconfig,
    [switch]$BuildOnly
)

# NOTE: do NOT use "Stop" here. ESP-IDF's export.ps1 writes progress text to
# stderr (e.g. "Activating ESP-IDF 5.4"), and PowerShell 5.1 in Stop mode turns
# any native-command stderr into a terminating error. We check $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"

# ---------------------------------------------------------------- locations
$IdfPath  = "D:\Run_Tao\esp-idf"
# This script sits at the project root (DialectVoice_SZPI\), so PSScriptRoot IS the root.
$RepoRoot = $PSScriptRoot
$ProjDir  = Join-Path $RepoRoot "firmware"

if (-not (Test-Path $IdfPath))            { throw "ESP-IDF not found: $IdfPath" }
if (-not (Test-Path $ProjDir))            { throw "firmware dir not found: $ProjDir" }
if (-not (Test-Path (Join-Path $IdfPath "export.ps1"))) { throw "export.ps1 missing in $IdfPath" }

# ------------------------------------------------------- enter IDF environment
# Pre-flight: ESP-IDF's export.ps1 requires the 'rich' module in its own venv.
# If missing, export.ps1 aborts with "Unable to import the rich module".
$IdfEnvs = Join-Path $env:USERPROFILE ".espressif\python_env"
if (Test-Path $IdfEnvs) {
    Get-ChildItem $IdfEnvs -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $cand = Join-Path $_.FullName "Scripts\python.exe"
        if (Test-Path $cand) {
            & $cand -c "import rich" 2>$null
            if ($LASTEXITCODE -ne 0) {
                Write-Host "      repairing: installing 'rich' into $($_.Name) ..." -ForegroundColor Yellow
                & $cand -m pip install rich --disable-pip-version-check 2>&1 | Out-Null
            }
        }
    }
}

Write-Host "[1/5] Importing ESP-IDF environment ($IdfPath) ..." -ForegroundColor Cyan
. (Join-Path $IdfPath "export.ps1") | Out-Null
if (-not (Get-Command idf.py -ErrorAction SilentlyContinue)) {
    throw "idf.py still unavailable after export.ps1 -- check IDF installation."
}

Set-Location $ProjDir
Write-Host "[2/5] Project: $ProjDir" -ForegroundColor Cyan

# ------------------------------------------------------------- optional clean
if ($Reconfig) {
    Write-Host "      -Reconfig: removing sdkconfig / sdkconfig.old" -ForegroundColor Yellow
    Remove-Item -Force -ErrorAction SilentlyContinue "sdkconfig", "sdkconfig.old"
}
if ($Clean) {
    Write-Host "      -Clean: idf.py fullclean" -ForegroundColor Yellow
    idf.py fullclean
}

# --------------------------------------------------------------- target check
if (-not (Test-Path "sdkconfig")) {
    Write-Host "[3/5] First build: set-target esp32s3" -ForegroundColor Cyan
    idf.py set-target esp32s3
} else {
    Write-Host "[3/5] sdkconfig exists, keep target" -ForegroundColor Cyan
}

# --------------------------------------------------------------- menuconfig
if ($Menuconfig) {
    idf.py menuconfig
    exit 0
}

# --------------------------------------------------------------------- build
Write-Host "[4/5] Building ..." -ForegroundColor Cyan
idf.py build
if ($LASTEXITCODE -ne 0) { throw "build failed (exit $LASTEXITCODE)" }

if ($BuildOnly) {
    Write-Host "Build finished (-BuildOnly). Binary: $ProjDir\build\*.bin" -ForegroundColor Green
    exit 0
}

# --------------------------------------------------------------------- flash
$flashArgs = @("flash")
if ($Port -ne "") { $flashArgs += @("-p", $Port) }

if ($Erase) {
    Write-Host "[5/5] Erasing whole flash then flashing ..." -ForegroundColor Yellow
    $eraseArgs = @("erase-flash")
    if ($Port -ne "") { $eraseArgs += @("-p", $Port) }
    idf.py @eraseArgs
    if ($LASTEXITCODE -ne 0) { throw "erase-flash failed" }
} else {
    Write-Host "[5/5] Flashing ..." -ForegroundColor Cyan
}

idf.py @flashArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Flash failed. Checklist:" -ForegroundColor Red
    Write-Host "  1) Is the board plugged into the correct USB-C port? (try the other one)"
    Write-Host "  2) Does Device Manager show a COM port? (CH343 / USB-SERIAL / USB JTAG)"
    Write-Host "  3) Hold BOOT (GPIO0), tap RST, release BOOT -> then flash"
    Write-Host "  4) Close any serial monitor / Arduino IDE occupying the port"
    throw "flash failed"
}

Write-Host "Flash OK." -ForegroundColor Green

# ------------------------------------------------------------------- monitor
if ($Monitor) {
    $monArgs = @("monitor")
    if ($Port -ne "") { $monArgs += @("-p", $Port) }
    Write-Host "Opening monitor (Ctrl+] to quit) ..." -ForegroundColor Cyan
    idf.py @monArgs
}

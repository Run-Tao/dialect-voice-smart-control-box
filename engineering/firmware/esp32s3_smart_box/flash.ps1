param(
    [Parameter(Mandatory = $true)]
    [string]$Port,
    [ValidateSet("usb", "uart")]
    [string]$Transport = "usb",
    [int]$Baud = 921600
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$image = Join-Path $root "dist\merged-$Transport.bin"
if (-not (Test-Path -LiteralPath $image)) {
    throw "找不到固件: $image"
}

python -m esptool version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "缺少 esptool。先运行: python -m pip install esptool"
}

Write-Host "烧录 $image -> $Port"
python -m esptool --chip esp32s3 --port $Port --baud $Baud write_flash 0x0 $image
if ($LASTEXITCODE -ne 0) {
    throw "烧录失败。请确认开发板已进入下载模式，且 COM 口未被串口监视器占用。"
}

Write-Host "烧录完成。按 RESET 后以 115200 波特率打开串口。"

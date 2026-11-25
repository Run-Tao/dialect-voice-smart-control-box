# 编译开题报告材料（xelatex 跑两遍，生成目录和交叉引用）
# 需要 TeX Live（含 ctex / xelatex）。Windows 下默认把 D:\texlive\2026 加进 PATH，
# 其他机器请改成自己的 TeX Live 路径，或直接确保 xelatex 在 PATH 里。
$ErrorActionPreference = 'Stop'
if (-not (Get-Command xelatex -ErrorAction SilentlyContinue)) {
    $env:Path = "D:\texlive\2026\bin\windows;$env:Path"
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

foreach ($name in @('开题报告', '开题报告材料说明')) {
    Write-Host "=== $name ===" -ForegroundColor Cyan
    xelatex -interaction=nonstopmode -halt-on-error "$name.tex"
    if ($LASTEXITCODE -ne 0) { throw "$name 第一遍编译失败，看 $name.log" }
    xelatex -interaction=nonstopmode -halt-on-error "$name.tex"
    if ($LASTEXITCODE -ne 0) { throw "$name 第二遍编译失败，看 $name.log" }
}

# 清掉中间文件（保留 .pdf/.tex）
Remove-Item -Force *.aux, *.log, *.out, *.toc -ErrorAction SilentlyContinue
Write-Host "完成：开题报告.pdf / 开题报告材料说明.pdf" -ForegroundColor Green

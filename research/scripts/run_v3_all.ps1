# -*- coding: utf-8 -*-
"""v3 一键复现脚本（PowerShell）。

用法（项目根目录）::

    powershell -ExecutionPolicy Bypass -File scripts/run_v3_all.ps1
    powershell -ExecutionPolicy Bypass -File scripts/run_v3_all.ps1 -SkipTrain   # 复用已有模型

前置：主环境 python 具备 numpy；训练需要 torch；E1 基线加速建议 numba。
仅当需要「从零重建数据」时才需要联网跑 TTS（见 docs/data_v3.md / data_v3_1.md）。
"""
param(
    [switch]$SkipTrain,
    [switch]$SkipExperiments,
    [string]$RunName = "main_v3_1"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root
$env:PYTHONIOENCODING = "utf-8"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$Python = "D:\anaconda3\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

Write-Host "=== 0. 环境检查 ===" -ForegroundColor Cyan
& $Python -c "import numpy, sys; print('numpy', numpy.__version__, 'python', sys.version.split()[0])"
& $Python -c "import json; c=json.load(open('config_v3.json',encoding='utf-8')); print('数据:', c['data']['manifest'], '+', c['data']['split_file'])"

Write-Host "=== 1. 单元测试（全量） ===" -ForegroundColor Cyan
& $Python -m unittest discover -s tests

if (-not $SkipTrain) {
    Write-Host "=== 2. 训练 3 种子 CNN（仅验证集选择） ===" -ForegroundColor Cyan
    & $Python -u scripts/train_v3_model.py --threads 16
} else {
    Write-Host "=== 2. 跳过训练，复用 data/models/v3_cnn_ensemble.npz ===" -ForegroundColor Yellow
}

if (-not $SkipExperiments) {
    Write-Host "=== 3. 正式对照实验 E1–E10 ===" -ForegroundColor Cyan
    & $Python -u scripts/run_v3_experiments.py --run-name $RunName

    Write-Host "=== 4. 拒识评估（阈值仅验证集+校准集） ===" -ForegroundColor Cyan
    & $Python -u scripts/evaluate_v3_rejection.py --out "outputs/v3_experiments/$RunName/rejection.json"

    Write-Host "=== 5. 生成报告表格 ===" -ForegroundColor Cyan
    & $Python scripts/report_v3_tables.py $RunName | Out-File -Encoding utf8 "outputs/v3_tables_$RunName.md"
}

Write-Host "=== 6. 端到端演示（CLI，7 条命令） ===" -ForegroundColor Cyan
& $Python scripts/demo_v3.py --sequence --model data/models/v3_cnn_ensemble.npz

Write-Host "完成：模型 data/models/v3_cnn_ensemble.npz；结果 outputs/v3_experiments/$RunName/；表格 outputs/v3_tables_$RunName.md" -ForegroundColor Green

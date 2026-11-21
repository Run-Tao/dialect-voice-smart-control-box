@echo off
rem 方言智居 CLI 示例：识别一个 wav 文件
cd /d %~dp0
set PYTHONUTF8=1
.venv\Scripts\python.exe src\dialect_home\cli.py --wav data\corpus\yue01.wav
pause
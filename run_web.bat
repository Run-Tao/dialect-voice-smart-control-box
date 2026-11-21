@echo off
rem 方言智居 Web 服务启动器
cd /d %~dp0
set PYTHONUTF8=1
.venv\Scripts\python.exe src\dialect_home\server.py
pause
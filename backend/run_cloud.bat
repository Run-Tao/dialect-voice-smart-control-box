@echo off
rem 云端服务启动脚本（PC 版引擎适配层）
set PYTHONUTF8=1
cd /d %~dp0
"D:\Projects\dialect-smart-home\.venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
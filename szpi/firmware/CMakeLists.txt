cmake_minimum_required(VERSION 3.16)

# 立创·实战派 ESP32-S3 方言语音端云协同项目
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

# 显式声明 C / CXX / ASM，**不要**让 CMake 自动探测 RC（Windows 资源编译器）。
# 不加这行时，CMake 3.30 在 Windows 上会尝试启用 RC 语言，而 xtensa 工具链
# 不含 windres，导致 configure 阶段报：
#   "The test project needs language RC which is not enabled."
#   "CMAKE_RC_COMPILER not set, after EnableLanguage"
project(dialect_voice_szpi C CXX ASM)

# ============================================================================
#  cmake/Toolchain-mingw.cmake — MinGW + SFML 的工具链定义
#
#  ★ 为什么需要单独一个工具链文件（而不是写在 CMakeLists.txt 里）
#
#    `-G "MinGW Makefiles"` 这个 generator 在【读 CMakeLists.txt 之前】
#    就要校验 `CMAKE_MAKE_PROGRAM`。所以写在 CMakeLists.txt 里太晚 ——
#    配 CMakeLists 时会直接死在：
#
#        CMake Error: CMake was unable to find a build program corresponding
#                     to "MinGW Makefiles".  CMAKE_MAKE_PROGRAM is not set.
#
#    ⚠️ 这个坑原来被 `build.bat` 掩盖着：它每次都用
#       `-DCMAKE_MAKE_PROGRAM=...` 显式传进来，所以"配 CMakeLists 却漏了它"
#       一直没被发现 —— 直到有人**直接调 cmake**（CI、或者只改了 CMakeLists）。
#
#    工具链文件是 CMake 的**标准机制**：它在 `project()` 之前被读，
#    正好赶上设置编译器与 make 程序。
#
#  ★ 怎么覆盖路径（按优先级）
#      ① 命令行：  cmake -S . -B build -G "MinGW Makefiles"
#                       -DCMAKE_TOOLCHAIN_FILE=cmake/Toolchain-mingw.cmake
#                       -DMINGW_ROOT=D:/mingw64 -DSFML_ROOT=D:/SFML-2.6.2
#      ② 环境变量：set MINGW_ROOT=D:\mingw64
#      ③ 都不给 → 默认 C:/mingw64 与 C:/SFML-2.6.2
#
#  ★ 用 build.bat 就不必操心这个文件 —— 脚本会自动探测并带上它。
# ============================================================================

# ---- 根目录：命令行 > 环境变量 > 默认 ----
if(NOT DEFINED MINGW_ROOT)
    if(DEFINED ENV{MINGW_ROOT})
        set(MINGW_ROOT "$ENV{MINGW_ROOT}")
    else()
        set(MINGW_ROOT "C:/mingw64")
    endif()
endif()
if(NOT DEFINED SFML_ROOT)
    if(DEFINED ENV{SFML_ROOT})
        set(SFML_ROOT "$ENV{SFML_ROOT}")
    else()
        set(SFML_ROOT "C:/SFML-2.6.2")
    endif()
endif()

# ---- 编译器与 make（存在才钉死，否则交给 CMake 去 PATH 里找）----
if(EXISTS "${MINGW_ROOT}/bin/gcc.exe")
    set(CMAKE_C_COMPILER   "${MINGW_ROOT}/bin/gcc.exe"   CACHE FILEPATH "" FORCE)
    set(CMAKE_CXX_COMPILER "${MINGW_ROOT}/bin/g++.exe"   CACHE FILEPATH "" FORCE)
endif()
if(EXISTS "${MINGW_ROOT}/bin/mingw32-make.exe")
    set(CMAKE_MAKE_PROGRAM "${MINGW_ROOT}/bin/mingw32-make.exe" CACHE FILEPATH "" FORCE)
endif()

# ---- 把两个根目录也存进 cache，供 CMakeLists.txt 使用（DLL 拷贝要用）----
set(MINGW_ROOT "${MINGW_ROOT}" CACHE PATH "MinGW 安装根目录" FORCE)
set(SFML_ROOT  "${SFML_ROOT}"  CACHE PATH "SFML 安装根目录"  FORCE)

# Windows 目标
set(CMAKE_SYSTEM_NAME Windows)

message(STATUS "[toolchain] MinGW = ${MINGW_ROOT}")
message(STATUS "[toolchain] SFML  = ${SFML_ROOT}")

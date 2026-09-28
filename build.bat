@echo off
setlocal enabledelayedexpansion

REM ===========================================================================
REM  build.bat - one-click build (7 targets)
REM
REM  Paths are OVERRIDABLE - nothing is hard-coded:
REM      set CMAKE_EXE=D:\CMake\bin\cmake.exe
REM      set MINGW_ROOT=D:\mingw64
REM      set SFML_ROOT=D:\SFML-2.6.2
REM      build.bat
REM
REM  If unset, the script auto-detects cmake / g++ from PATH.
REM  SFML must be given (it has no standard install location).
REM
REM  *** IMPORTANT: KEEP THIS FILE PURE ASCII ***
REM  cmd.exe reads .bat files using the OEM code page (936/GBK on Chinese
REM  Windows). UTF-8 Chinese comments get mis-decoded into garbage and are
REM  then executed as commands - the build dies with bizarre errors like
REM  "'xxx' is not recognized as an internal or external command".
REM  This bit us once. Do not put non-ASCII text in .bat / .cmd files.
REM
REM  *** PORTABILITY: the whole folder may be COPIED ANYWHERE ***
REM  CMakeCache.txt stores the ABSOLUTE source path it was created with.
REM  If the folder is moved/copied with build\ included, re-running cmake
REM  fails with:
REM      The current CMakeCache.txt directory ... is different than the
REM      directory ... where CMakeCache.txt was created
REM      The source ... does not match the source ... used to generate cache
REM  Step 0 below detects exactly that case and wipes the stale cache.
REM  Without it the user has to know to delete build\ by hand.
REM ===========================================================================

REM ---- 1) locate cmake ----
if not defined CMAKE_EXE (
    if exist "C:\Program Files\CMake\bin\cmake.exe" (
        set "CMAKE_EXE=C:\Program Files\CMake\bin\cmake.exe"
    ) else (
        for /f "delims=" %%i in ('where cmake 2^>nul') do (
            if not defined CMAKE_EXE set "CMAKE_EXE=%%i"
        )
    )
)
if not defined CMAKE_EXE (
    echo ERROR: cmake not found.
    echo        Install it, or set CMAKE_EXE to cmake.exe
    echo        e.g.  set CMAKE_EXE=D:\CMake\bin\cmake.exe
    pause
    exit /b 1
)

REM ---- 2) locate MinGW ----
if not defined MINGW_ROOT (
    if exist "C:\mingw64\bin\g++.exe" (
        set "MINGW_ROOT=C:\mingw64"
    ) else (
        for /f "delims=" %%i in ('where g++ 2^>nul') do (
            if not defined GXX_PATH set "GXX_PATH=%%i"
        )
        if defined GXX_PATH (
            REM strip \bin\g++.exe  ->  root
            for %%d in ("!GXX_PATH!\..\..") do set "MINGW_ROOT=%%~fd"
        )
    )
)

REM ---- 3) locate SFML ----
if not defined SFML_ROOT (
    if exist "C:\SFML-2.6.2\lib\cmake\SFML" set "SFML_ROOT=C:\SFML-2.6.2"
)
if not defined SFML_ROOT (
    echo ERROR: SFML 2.6 not found.
    echo        Download the MinGW build of SFML 2.6.x and unzip it, then:
    echo            set SFML_ROOT=D:\SFML-2.6.2
    pause
    exit /b 1
)

set "SRC_DIR=%~dp0."
set "BUILD_DIR=%~dp0build"

REM Toolchain file MUST be an absolute path: cmake treats a relative value as a
REM module name and fails with "Could not find toolchain file: cmake/...".
set "TOOLCHAIN=%~dp0cmake\Toolchain-mingw.cmake"
if not exist "%TOOLCHAIN%" (
    echo ERROR: toolchain file not found: %TOOLCHAIN%
    pause
    exit /b 1
)

echo === Snake NN Lab - Build ===
echo   CMake : %CMAKE_EXE%
if defined MINGW_ROOT (echo   MinGW : %MINGW_ROOT%) else (echo   MinGW : from PATH)
echo   SFML  : %SFML_ROOT%
echo   Src   : %SRC_DIR%
echo.

REM ---- 0) reject a cache created in ANOTHER directory ----
REM CMakeCache.txt records the source dir it was generated from, on the line
REM     CMAKE_HOME_DIRECTORY:INTERNAL=<abs path, forward slashes>
REM If that value is missing or disagrees with our source dir, the cache is
REM stale (the folder was moved or copied with build\ included) and must go.
set "STALE="
if exist "%BUILD_DIR%\CMakeCache.txt" (
    set "CACHED_SRC="
    for /f "tokens=1,* delims==" %%a in ('findstr /b /c:"CMAKE_HOME_DIRECTORY:" "%BUILD_DIR%\CMakeCache.txt" 2^>nul') do set "CACHED_SRC=%%b"

    if not defined CACHED_SRC (
        echo [0/2] CMakeCache.txt records no source directory - discarding it.
        set "STALE=1"
    ) else (
        REM Canonicalise both sides for a fair comparison:
        REM   - "%~dp0." leaves a trailing dot  ->  C:\path\.   (canonicalise via %%~fd)
        REM   - CMake writes forward slashes   ->  C:/path
        for %%d in ("%SRC_DIR%") do set "WANT_NORM=%%~fd"
        set "WANT_NORM=!WANT_NORM:\=/!"
        set "CACHED_NORM=!CACHED_SRC:\=/!"
        if /i "!CACHED_NORM!"=="!WANT_NORM!" (
            REM cache belongs to this directory -> keep it (incremental build)
        ) else (
            echo [0/2] STALE CMake cache detected - this folder was moved or copied.
            echo         cache generated in : !CACHED_SRC!
            echo         building in        : !WANT_NORM!
            echo         discarding the cache and reconfiguring from scratch.
            set "STALE=1"
        )
    )
)

if defined STALE (
    rmdir /s /q "%BUILD_DIR%" 2>nul
    if exist "%BUILD_DIR%\CMakeCache.txt" (
        echo ERROR: could not remove %BUILD_DIR%\CMakeCache.txt
        echo        Close any running snake_game.exe / console tool, then retry.
        pause
        exit /b 1
    )
    echo         cache cleared.
    echo.
)

if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

echo [1/2] Configuring CMake...
"%CMAKE_EXE%" -S "%SRC_DIR%" -B "%BUILD_DIR%" ^
  -G "MinGW Makefiles" ^
  -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DMINGW_ROOT="%MINGW_ROOT%" ^
  -DSFML_ROOT="%SFML_ROOT%"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo [2/2] Building...
"%CMAKE_EXE%" --build "%BUILD_DIR%"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo === Build OK ===
echo   build\snake_game.exe         game
echo   build\nn_selftest.exe        MLP self-test
echo   build\bench_sim.exe          perf probe
echo   build\train_probe.exe        CLI training
echo   build\replay_check.exe       world fingerprint regression
echo   build\bandit_check.exe       learning-chain self-proof
echo   build\statebandit_check.exe  conditional-policy self-proof
echo.
echo Double-click run.bat to start the game.
pause

@echo off
REM ===========================================================================
REM  run.bat - launch the game
REM
REM  *** KEEP THIS FILE PURE ASCII ***
REM  cmd.exe reads .bat using the OEM code page; UTF-8 non-ASCII text gets
REM  mis-decoded into garbage and executed as commands. This bit us once.
REM
REM  All paths are derived from %~dp0 (this script's own folder), so the
REM  script works no matter which directory it is called from - and the whole
REM  project folder can be moved anywhere.
REM
REM  *** WHY THE `cd /d` BELOW IS REQUIRED ***
REM  `start` does NOT change the child's working directory - the child inherits
REM  the caller's cwd. An earlier version of this file claimed the opposite and
REM  relied on it, which broke every relative path when the script was launched
REM  from anywhere except build\ (double-clicking run.bat from Explorer makes
REM  the cwd the PROJECT ROOT):
REM      saves/                        -> project-root\saves\   (wrong folder)
REM      assets/picture/cathead.png    -> not found
REM      train_metrics.csv             -> "cannot read training data"
REM  Setting the cwd explicitly to build\ makes the behaviour deterministic and
REM  identical no matter how the game is started.
REM ===========================================================================

set "EXE=%~dp0build\snake_game.exe"

if not exist "%EXE%" (
    echo ERROR: %EXE% not found.
    echo        Run build.bat first.
    pause
    exit /b 1
)

REM Run the game with build\ as the working directory (see the note above).
cd /d "%~dp0build"

echo Starting Snake Game...
start "" "%EXE%"

@echo off
rem Opens the sound browser on your library. Close this window to stop it.
rem Drag your library folder onto this file, or run it and type the folder once: it is remembered in library.txt.
setlocal
set "HERE=%~dp0"
call "%HERE%tools\library.bat" "%~1" || exit /b 1
python "%HERE%skills\audio-designer\scripts\sound_browser.py" --root "%LIBRARY%"
set "RUN_STATUS=%errorlevel%"
pause
exit /b %RUN_STATUS%

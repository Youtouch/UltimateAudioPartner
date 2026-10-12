@echo off
rem Opens the sound browser on your library. Close this window to stop it.
rem Drag your library folder onto this file, or run it and type the folder once: it is remembered in library.txt.
setlocal
set "HERE=%~dp0"
if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help
if /i "%~1"=="demo" goto :demo
call "%HERE%tools\library.bat" "%~1" || exit /b 1
python "%HERE%skills\audio-designer\scripts\sound_browser.py" --root "%LIBRARY%"
set "RUN_STATUS=%errorlevel%"
pause
exit /b %RUN_STATUS%

:demo
python "%HERE%tools\demo.py"
exit /b %errorlevel%

:help
echo Browse, audition, review and export sounds from a tagged audio library.
echo Usage: "Sound Browser.bat" [library folder ^| demo ^| --help]
echo Arguments:
echo   library folder  Optional audio folder; omitted uses library.txt or AUDIO_LIBRARY, then asks.
echo   demo            Generate a temporary example library and open the browser.
echo   --help, -h      Print this help and exit.
echo Example: "Sound Browser.bat" "D:\MySounds"
exit /b 0

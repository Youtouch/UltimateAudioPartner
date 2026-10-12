@echo off
rem Sets LIBRARY for the launchers: a dragged-in folder, else library.txt, else AUDIO_LIBRARY, else asks once.
set "STORE=%~dp0..\library.txt"
set "LIBRARY=%~1"
if "%LIBRARY%"=="" if exist "%STORE%" set /p LIBRARY=<"%STORE%"
if "%LIBRARY%"=="" set "LIBRARY=%AUDIO_LIBRARY%"
if "%LIBRARY%"=="" set /p "LIBRARY=Folder holding your sound library: "
set "LIBRARY=%LIBRARY:"=%"
if not exist "%LIBRARY%\" (
    echo Not a folder: "%LIBRARY%"
    pause
    exit /b 1
)
for %%I in ("%LIBRARY%") do set "LIBRARY=%%~fI"
rem Delayed expansion writes metacharacters such as & as folder text.
setlocal EnableDelayedExpansion
> "!STORE!" echo(!LIBRARY!
if errorlevel 1 (
    endlocal
    exit /b 1
)
endlocal
exit /b 0

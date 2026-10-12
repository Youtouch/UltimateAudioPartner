@echo off
rem Tags every audio file in your library (what each one contains, plus a fingerprint for search),
rem measures each one (length, loudness, attack, brightness...) and finds duplicates and variation sets.
rem Resumable: close the window any time and run it again to continue, or to add new files later.
rem Drag your library folder onto this file, or run it and type the folder once: it is remembered in library.txt.
setlocal
set "HERE=%~dp0"
if /i "%~1"=="--help" goto :help
if /i "%~1"=="-h" goto :help
if /i "%~1"=="demo" goto :demo
call "%HERE%tools\library.bat" "%~1" || exit /b 1
python "%HERE%skills\audio-designer\scripts\audio_tag.py" tag --root "%LIBRARY%" --out "%LIBRARY%\AUDIO_TAGS.csv" || goto :failed
python "%HERE%skills\audio-designer\scripts\audio_search.py" names --root "%LIBRARY%" || goto :failed
python "%HERE%skills\audio-designer\scripts\audio_describe.py" describe --root "%LIBRARY%" || goto :failed
python "%HERE%skills\audio-designer\tagging\vocab_tags.py" build --root "%LIBRARY%" || goto :failed
python "%HERE%skills\audio-designer\scripts\sfx_groups.py" build --root "%LIBRARY%" || goto :failed
pause
exit /b 0

:failed
set "RUN_STATUS=%errorlevel%"
pause
exit /b %RUN_STATUS%

:demo
python "%HERE%tools\demo.py"
exit /b %errorlevel%

:help
echo Tag and measure an audio library, then group duplicate recordings and variations.
echo Usage: "Tag Library.bat" [library folder ^| demo ^| --help]
echo Arguments:
echo   library folder  Optional audio folder; omitted uses library.txt or AUDIO_LIBRARY, then asks.
echo   demo            Generate a temporary example library and open the browser.
echo   --help, -h      Print this help and exit.
echo Example: "Tag Library.bat" "D:\MySounds"
exit /b 0

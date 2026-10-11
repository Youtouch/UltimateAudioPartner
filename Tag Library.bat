@echo off
rem Tags every audio file in your library (what each one contains, plus a fingerprint for search).
rem Resumable: close the window any time and run it again to continue, or to add new files later.
rem Drag your library folder onto this file, or run it and type the folder once: it is remembered in library.txt.
setlocal
set "HERE=%~dp0"
call "%HERE%tools\library.bat" "%~1" || goto :eof
python "%HERE%skills\audio-designer\scripts\audio_tag.py" tag --root "%LIBRARY%" --out "%LIBRARY%\AUDIO_TAGS.csv"
python "%HERE%skills\audio-designer\scripts\audio_search.py" names --root "%LIBRARY%"
python "%HERE%skills\audio-designer\tagging\vocab_tags.py" build --root "%LIBRARY%"
pause

#!/usr/bin/env bash
# Opens the sound browser on your library. Ctrl+C stops it.
# Usage: ./sound-browser.sh [library folder]   (asked once and remembered in library.txt if omitted)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/tools/library.sh"
python3 "$HERE/skills/audio-designer/scripts/sound_browser.py" --root "$LIBRARY"

#!/usr/bin/env bash
# Opens the sound browser on your library. Ctrl+C stops it.
# Usage: ./sound-browser.sh [library folder]   (asked once and remembered in library.txt if omitted)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    cat <<'HELP'
Browse, audition, review and export sounds from a tagged audio library.
Usage: ./sound-browser.sh [library folder | demo | --help]
Arguments:
  library folder  Optional audio folder; omitted uses library.txt or AUDIO_LIBRARY, then asks.
  demo            Generate a temporary example library and open the browser.
  --help, -h      Print this help and exit.
Example: ./sound-browser.sh "/path/to/MySounds"
HELP
    exit 0
fi
if [ "${1:-}" = demo ]; then
    exec python3 "$HERE/tools/demo.py"
fi
. "$HERE/tools/library.sh"
python3 "$HERE/skills/audio-designer/scripts/sound_browser.py" --root "$LIBRARY"

#!/usr/bin/env bash
# Tags every audio file in your library (what each one contains, plus a fingerprint for search),
# measures each one (length, loudness, attack, brightness...) and finds duplicates and variation sets.
# Resumable: stop any time and run it again to continue, or to add new files later.
# Usage: ./tag-library.sh [library folder]   (asked once and remembered in library.txt if omitted)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "${1:-}" = --help ] || [ "${1:-}" = -h ]; then
    cat <<'HELP'
Tag and measure an audio library, then group duplicate recordings and variations.
Usage: ./tag-library.sh [library folder | demo | --help]
Arguments:
  library folder  Optional audio folder; omitted uses library.txt or AUDIO_LIBRARY, then asks.
  demo            Generate a temporary example library and open the browser.
  --help, -h      Print this help and exit.
Example: ./tag-library.sh "/path/to/MySounds"
HELP
    exit 0
fi
if [ "${1:-}" = demo ]; then
    exec python3 "$HERE/tools/demo.py"
fi
. "$HERE/tools/library.sh"
python3 "$HERE/skills/audio-designer/scripts/audio_tag.py" tag --root "$LIBRARY" --out "$LIBRARY/AUDIO_TAGS.csv"
python3 "$HERE/skills/audio-designer/scripts/audio_search.py" names --root "$LIBRARY"
python3 "$HERE/skills/audio-designer/scripts/audio_describe.py" describe --root "$LIBRARY"
python3 "$HERE/skills/audio-designer/tagging/vocab_tags.py" build --root "$LIBRARY"
python3 "$HERE/skills/audio-designer/scripts/sfx_groups.py" build --root "$LIBRARY"

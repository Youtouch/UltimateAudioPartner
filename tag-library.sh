#!/usr/bin/env bash
# Tags every audio file in your library (what each one contains, plus a fingerprint for search).
# Resumable: stop any time and run it again to continue, or to add new files later.
# Usage: ./tag-library.sh [library folder]   (asked once and remembered in library.txt if omitted)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/tools/library.sh"
python3 "$HERE/skills/audio-designer/scripts/audio_tag.py" tag --root "$LIBRARY" --out "$LIBRARY/AUDIO_TAGS.csv"
python3 "$HERE/skills/audio-designer/scripts/audio_search.py" names --root "$LIBRARY"
python3 "$HERE/skills/audio-designer/tagging/vocab_tags.py" build --root "$LIBRARY"

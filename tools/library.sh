# Sourced by the launchers: sets LIBRARY from $1, else library.txt, else AUDIO_LIBRARY, else asks once.
STORE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/library.txt"
LIBRARY="${1:-}"
[ -z "$LIBRARY" ] && [ -f "$STORE" ] && LIBRARY="$(head -n 1 "$STORE")"
# A folder saved by the Windows launcher can have a CRLF line ending.
LIBRARY="${LIBRARY%$'\r'}"
[ -z "$LIBRARY" ] && LIBRARY="${AUDIO_LIBRARY:-}"
[ -z "$LIBRARY" ] && read -r -p "Folder holding your sound library: " LIBRARY
if [ ! -d "$LIBRARY" ]; then
    echo "Not a folder: $LIBRARY"
    exit 1
fi
LIBRARY="$(cd -- "$LIBRARY" && pwd -P)"
printf '%s\n' "$LIBRARY" > "$STORE"

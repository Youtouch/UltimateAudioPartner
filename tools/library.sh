# Sourced by the launchers: sets LIBRARY from $1, else library.txt, else AUDIO_LIBRARY, else asks once.
STORE="$(dirname "$0")/library.txt"
LIBRARY="${1:-}"
[ -z "$LIBRARY" ] && [ -f "$STORE" ] && LIBRARY="$(head -n 1 "$STORE")"
[ -z "$LIBRARY" ] && LIBRARY="${AUDIO_LIBRARY:-}"
[ -z "$LIBRARY" ] && read -r -p "Folder holding your sound library: " LIBRARY
if [ ! -d "$LIBRARY" ]; then
    echo "Not a folder: $LIBRARY"
    exit 1
fi
printf '%s\n' "$LIBRARY" > "$STORE"

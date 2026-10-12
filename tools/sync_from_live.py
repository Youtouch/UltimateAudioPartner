"""Stage the allowlisted live skill, check privacy and tests, then publish the copy.

python tools/sync_from_live.py [--from ~/.agents/skills/audio-designer] [--dest TEMP_REPO]
The destination is a repository root. No files are replaced until every gate passes.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
import zlib

PUBLISH_TAGGING = {'vocab_tags.py', 'self_tests.py', 'label_rules.py', 'label_self_tests.py', 'label_decisions_v2.json',
                   'label_decisions_v3.json', 'game_vocab.py', 'vocabulary_synonyms.csv'}
PUBLISH_RENDER = {'render_mix.py', 'render_tests.py', 'README.md'}
PUBLISH_MIX = {'mix_check.py', 'calibration_tests.py', 'README.md'}
PUBLISH_BENCH = {'bench_select.py', 'bench_eval.py', 'bench_clap.py', 'methods.py', 'tagger_trial.py', 'short_sfx.py', 'README.md',
                 'vocabulary.csv', 'queries.csv'}
PUBLISH_ROOT = {'SKILL.md', 'references', 'scripts', 'bench', 'tagging', 'mix', 'render'}
PUBLISH_SCRIPTS = {'audio_tag.py', 'audio_map.py', 'audio_index.py', 'audio_search.py',
                   'sound_browser.py', 'sound_browser.html', 'library_metadata.py',
                   'library_tests.py', 'browser_tests.py', 'audio_describe.py', 'sfx_groups.py', 'export_sfx.py',
                   'audio_visuals.py', 'visual_tests.py', 'demo_metadata.py'}
PUBLISH_REFERENCES = {'mix-plan.md', 'music-and-beats.md', 'recognise-and-select.md', 'sound-without-hearing.md'}
REPO_FILES = {'README.md', 'CHANGELOG.md', 'NOTICE', 'LICENSE', 'requirements.txt', '.gitignore',
              'Tag Library.bat', 'Sound Browser.bat', 'tag-library.sh', 'sound-browser.sh'}
PUBLISH_IMAGES = {'library.png', 'sound-detail.png', 'explain.png', 'compare.png', 'export.png'}
PUBLISH_SCREENSHOT_FILES = PUBLISH_IMAGES | {'capture-evidence.json'}


def ignore(folder, names):
    out = {n for n in names if n == '__pycache__' or n.endswith(('.pyc', '.pyo'))}
    allowed = {'audio-designer': PUBLISH_ROOT, 'scripts': PUBLISH_SCRIPTS, 'references': PUBLISH_REFERENCES, 'bench': PUBLISH_BENCH,
               'tagging': PUBLISH_TAGGING, 'render': PUBLISH_RENDER, 'mix': PUBLISH_MIX}.get(Path(folder).name)
    if allowed is not None:
        out |= set(names) - allowed
    return out


def copy_skill(src, dst):
    """The single publication allowlist, also used by the fresh-clone smoke test."""
    source = Path(src).resolve()
    def selected(folder, names):
        if Path(folder).resolve() == source:
            return set(names) - PUBLISH_ROOT
        return ignore(folder, names)
    shutil.copytree(source, dst, ignore=selected)


def copy_repo_files(src, dst):
    """Only public source files; private word lists, library data and reports stay local."""
    dst.mkdir(parents=True, exist_ok=True)
    for name in sorted(REPO_FILES):
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)
    for folder, suffixes in [('tools', {'.py', '.sh', '.bat'}), ('examples', {'.csv', '.md'}), ('docs', {'.md'})]:
        if not (src / folder).is_dir():
            continue
        for p in sorted((src / folder).iterdir()):
            if p.is_file() and p.suffix in suffixes:
                target = dst / folder / p.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)
    for name in sorted(PUBLISH_SCREENSHOT_FILES):
        source = src / 'docs/img' / name
        if source.is_file():
            target = dst / 'docs/img' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def png_metadata(path):
    """Validate a PNG container and extract its text without scanning compressed pixels."""
    body = path.read_bytes()
    if body[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Invalid PNG signature')
    offset, texts, seen = 8, [], set()
    while offset < len(body):
        if offset + 12 > len(body):
            raise ValueError('Truncated PNG chunk')
        length = struct.unpack('>I', body[offset:offset + 4])[0]
        kind = body[offset + 4:offset + 8]
        end = offset + 12 + length
        if end > len(body):
            raise ValueError('Truncated PNG data')
        data = body[offset + 8:end - 4]
        if zlib.crc32(kind + data) != struct.unpack('>I', body[end - 4:end])[0]:
            raise ValueError('Invalid PNG checksum')
        seen.add(kind)
        if kind == b'tEXt':
            texts.append(data.replace(b'\0', b' ').decode('latin-1'))
        elif kind in (b'zTXt', b'iCCP'):
            keyword, rest = data.split(b'\0', 1)
            if not rest or rest[0] != 0:
                raise ValueError('Invalid compressed PNG text')
            texts.append(keyword.decode('latin-1') + ' ' + zlib.decompress(rest[1:]).decode('latin-1'))
        elif kind == b'iTXt':
            keyword, rest = data.split(b'\0', 1)
            flag, method = rest[:2]
            language, rest = rest[2:].split(b'\0', 1)
            translated, text = rest.split(b'\0', 1)
            if flag not in (0, 1) or method != 0:
                raise ValueError('Invalid international PNG text')
            if flag:
                text = zlib.decompress(text)
            texts.append(b' '.join((keyword, language, translated, text)).decode('utf-8', errors='replace'))
        elif kind == b'eXIf':
            texts.append(data.replace(b'\0', b'').decode('utf-8', errors='replace'))
        elif kind not in {b'IHDR', b'PLTE', b'IDAT', b'IEND', b'tRNS', b'sRGB', b'gAMA', b'cHRM', b'pHYs', b'bKGD', b'sBIT', b'tIME'}:
            raise ValueError('Unsupported PNG metadata')
        offset = end
        if kind == b'IEND':
            if offset != len(body):
                raise ValueError('Trailing PNG data')
            break
    if not {b'IHDR', b'IDAT', b'IEND'} <= seen:
        raise ValueError('Incomplete PNG')
    return '\n'.join(texts)


def privacy_check(root, words_path):
    """Reject literal private phrases, case-insensitively; never echo their matching lines."""
    if not words_path.is_file():
        print('NOTE privacy: no local tools/private_words.txt; only the publication allowlist applies', flush=True)
        return []
    words = [line.strip() for line in words_path.read_text(encoding='utf-8-sig').splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    failures = []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or '.git' in p.parts or p.name == 'private_words.txt':
            continue
        if p.suffix.lower() == '.png':
            # Compressed pixel bytes can accidentally spell short private tokens. Read actual
            # PNG text (including compressed zTXt/iTXt and EXIF) rather than interpreting pixels
            # as prose. Invalid/renamed files fail closed instead of bypassing the gate.
            try:
                content = png_metadata(p)
            except (OSError, ValueError, SyntaxError, zlib.error, struct.error):
                failures.append((p.relative_to(root).as_posix(), 0, 'invalid PNG'))
                continue
        else:
            content = p.read_text(encoding='utf-8', errors='replace')
        for number, line in enumerate(content.splitlines(), 1):
            for word in words:
                if word.casefold() in line.casefold():
                    failures.append((p.relative_to(root).as_posix(), number, word))
    for name, number, word in failures:
        print(f'FAIL privacy {name}:{number}: blocked phrase {word!r}', flush=True)
    print(f'{"FAIL" if failures else "PASS"} privacy gate ({len(words)} private phrases)', flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     epilog='Example: python tools/sync_from_live.py --dest TEMP_REPO',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--from', dest='src', type=Path, default=Path.home() / '.agents/skills/audio-designer',
                        help='live skill folder to stage')
    parser.add_argument('--dest', '--to', dest='dest', type=Path, help='destination repository root (default this checkout)')
    a = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    destination = (a.dest or repo).resolve()
    if not a.src.is_dir():
        parser.error(f'No live skill at {a.src}')
    with tempfile.TemporaryDirectory(prefix='uap-publish-') as temporary:
        stage = Path(temporary) / 'repo'
        copy_repo_files(repo, stage)
        copy_skill(a.src, stage / 'skills/audio-designer')
        if privacy_check(stage, repo / 'tools/private_words.txt'):
            return 1
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
        run = subprocess.run([sys.executable, '-B', str(stage / 'tools/check_all.py'), '--published'], cwd=stage, env=env)
        if run.returncode:
            print('FAIL sync: staged tests failed; destination was left untouched', flush=True)
            return 1
        destination.mkdir(parents=True, exist_ok=True)
        if destination != repo:
            copy_repo_files(stage, destination)
        dst = destination / 'skills/audio-designer'
        if not dst.resolve().is_relative_to(destination):
            raise ValueError('Skill destination resolves outside the selected repository')
        if dst.exists():
            shutil.rmtree(dst)
        copy_skill(stage / 'skills/audio-designer', dst)
        print(f'PASS sync: tested public copy at {destination}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


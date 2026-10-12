"""Stage the allowlisted live skill, check privacy and tests, then publish the copy.

python tools/sync_from_live.py [--from ~/.agents/skills/audio-designer] [--dest TEMP_REPO]
The destination is a repository root. No files are replaced until every gate passes.
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

PUBLISH_TAGGING = {'vocab_tags.py', 'self_tests.py', 'label_rules.py', 'label_self_tests.py', 'label_decisions_v2.json', 'vocabulary_synonyms.csv'}
PUBLISH_RENDER = {'render_mix.py', 'render_tests.py', 'README.md'}
PUBLISH_MIX = {'mix_check.py', 'calibration_tests.py', 'README.md'}
PUBLISH_BENCH = {'bench_select.py', 'bench_eval.py', 'bench_clap.py', 'methods.py', 'tagger_trial.py', 'README.md', 'vocabulary.csv', 'queries.csv'}
PUBLISH_ROOT = {'SKILL.md', 'references', 'scripts', 'bench', 'tagging', 'mix', 'render'}
PUBLISH_SCRIPTS = {'audio_tag.py', 'audio_map.py', 'audio_index.py', 'audio_search.py',
                   'sound_browser.py', 'sound_browser.html', 'library_metadata.py',
                   'library_tests.py', 'browser_tests.py'}
PUBLISH_REFERENCES = {'mix-plan.md', 'music-and-beats.md', 'recognise-and-select.md', 'sound-without-hearing.md'}
REPO_FILES = {'README.md', 'NOTICE', 'LICENSE', 'requirements.txt', '.gitignore',
              'Tag Library.bat', 'Sound Browser.bat', 'tag-library.sh', 'sound-browser.sh'}


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
    for folder, suffixes in [('tools', {'.py', '.sh', '.bat'}), ('examples', {'.csv', '.md'})]:
        if not (src / folder).is_dir():
            continue
        for p in sorted((src / folder).iterdir()):
            if p.is_file() and p.suffix in suffixes:
                target = dst / folder / p.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, target)


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
        for number, line in enumerate(p.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            for word in words:
                if word.casefold() in line.casefold():
                    failures.append((p.relative_to(root).as_posix(), number, word))
    for name, number, word in failures:
        print(f'FAIL privacy {name}:{number}: blocked phrase {word!r}', flush=True)
    print(f'{"FAIL" if failures else "PASS"} privacy gate ({len(words)} private phrases)', flush=True)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--from', dest='src', type=Path, default=Path.home() / '.agents/skills/audio-designer')
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


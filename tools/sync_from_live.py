"""Copy the installed audio-designer skill into this repo, leaving out library data.

    python tools/sync_from_live.py [--from ~/.agents/skills/audio-designer]

The installed skill is the one that gets edited; this repo is its public copy. The benchmark folder
holds data drawn from the user's own library (paths, snapshots, scores): only its tools are published.
"""

import argparse
import shutil
from pathlib import Path

PUBLISH_BENCH = {'bench_select.py', 'bench_eval.py', 'bench_clap.py', 'methods.py', 'tagger_trial.py', 'README.md', 'vocabulary.csv', 'queries.csv'}


def ignore(folder, names):
    out = {n for n in names if n == '__pycache__'}
    if Path(folder).name == 'bench':
        out |= {n for n in names if n not in PUBLISH_BENCH}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--from', dest='src', type=Path, default=Path.home() / '.agents/skills/audio-designer')
    a = parser.parse_args()
    dst = Path(__file__).resolve().parents[1] / 'skills' / 'audio-designer'
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(a.src, dst, ignore=ignore)
    for p in sorted(dst.rglob('*')):
        if p.is_file():
            print(p.relative_to(dst).as_posix())


if __name__ == '__main__':
    main()

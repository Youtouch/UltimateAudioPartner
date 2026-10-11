"""Copy the installed audio-designer skill into this repo, leaving out library data.

    python tools/sync_from_live.py [--from ~/.agents/skills/audio-designer]

The installed skill is the one that gets edited; this repo is its public copy. The benchmark folder
holds data drawn from the user's own library (paths, snapshots, scores): only its tools are published.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PUBLISH_TAGGING = {'vocab_tags.py', 'self_tests.py', 'label_rules.py', 'label_self_tests.py', 'label_decisions_v2.json', 'vocabulary_synonyms.csv'}
PUBLISH_MIX = {'mix_check.py', 'README.md'}
PUBLISH_BENCH = {'bench_select.py', 'bench_eval.py', 'bench_clap.py', 'methods.py', 'tagger_trial.py', 'README.md', 'vocabulary.csv', 'queries.csv'}


def ignore(folder, names):
    out = {n for n in names if n == '__pycache__'}
    if Path(folder).name == 'bench':
        out |= {n for n in names if n not in PUBLISH_BENCH}
    if Path(folder).name == 'tagging':
        out |= {n for n in names if n not in PUBLISH_TAGGING}
    if Path(folder).name == 'mix':
        out |= {n for n in names if n not in PUBLISH_MIX}
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
    # Another agent may be mid-edit in the live skill: a copied half-change shows up as a failing self-test here.
    # (audio_tag.py check is left out: it loads both models and takes minutes; run it by hand after tagger changes.)
    failed = []
    for script, cmd in [('scripts/audio_search.py', 'check'), ('scripts/audio_index.py', 'check'),
                        ('bench/bench_eval.py', 'check'), ('bench/tagger_trial.py', 'check'), ('tagging/vocab_tags.py', 'check'), ('mix/mix_check.py', 'check')]:
        if (dst / script).exists():
            run = subprocess.run([sys.executable, '-B', script, cmd], cwd=dst, capture_output=True, text=True)
            print(f'{"PASS" if run.returncode == 0 else "FAIL"} {script} {cmd}')
            if run.returncode:
                failed.append(script)
                print((run.stdout + run.stderr)[-800:])
    for cache in dst.rglob('__pycache__'):
        shutil.rmtree(cache)
    if failed:
        sys.exit(f'Self-tests failed in the copy: {", ".join(failed)}. Do not commit; is someone still editing the live skill?')


if __name__ == '__main__':
    main()

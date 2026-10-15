"""Run all UAP self-tests and a real 10-file fresh-clone smoke test.

python tools/check_all.py
--published checks exactly this repository; by default a maintainer's installed live skill is staged.
No dependencies are installed and every model call uses the existing offline cache.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from sync_from_live import copy_repo_files, copy_skill, privacy_check

# The publication gate imports no second list: it invokes this runner.
CHECKS = [
    ('scripts/audio_map.py', '--selftest'),
    ('scripts/audio_tag.py', 'check'),
    ('scripts/any_file_tests.py',),
    ('scripts/audio_search.py', 'check'),
    ('scripts/audio_index.py', 'check'),
    ('scripts/audio_describe.py', 'check'),
    ('scripts/sfx_groups.py', 'check'),
    ('scripts/export_sfx.py', 'check'),
    ('scripts/library_tests.py',),
    ('scripts/browser_tests.py',),
    ('scripts/visual_tests.py',),
    ('tagging/ucs.py', 'check'),
    ('tagging/recall_tags.py', 'check'),
    ('tagging/vocab_tags.py', 'check'),
    ('tagging/game_vocab.py', 'check'),
    ('mix/mix_check.py', 'check'),
    ('mix/calibration_tests.py',),
    ('render/render_mix.py', 'check'),
]
REPO_CHECKS = [('tools/launcher_tests.py',), ('tools/publication_tests.py',), ('tools/ui_tests.py',),
               ('tools/demo_tests.py',), ('tools/help_tests.py',), ('tools/polish_tests.py',)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                    epilog='Example: python tools/check_all.py --published',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--published', action='store_true', help='test this repository without looking for a live skill')
    parser.add_argument('--from', dest='source', type=Path, help='explicit live skill source')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    installed = Path.home() / '.agents/skills/audio-designer'
    source = args.source or (installed if not args.published and installed.is_dir() else repo / 'skills/audio-designer')
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    failures = []
    with tempfile.TemporaryDirectory(prefix='uap-check-') as temporary:
        env['NUMBA_CACHE_DIR'] = str(Path(temporary) / 'numba-cache')
        work = Path(temporary) / 'repo'
        copy_repo_files(repo, work)
        copy_skill(source, work / 'skills/audio-designer')
        print(f'Checking {"published" if args.published else "staged"} skill: {source}', flush=True)
        if privacy_check(work, repo / 'tools/private_words.txt'):
            failures.append('privacy')
        checks = [('skills/audio-designer/' + check[0], *check[1:]) for check in CHECKS] + REPO_CHECKS
        for check in checks:
            script, *arguments = check
            label = ' '.join(check)
            try:
                run = subprocess.run([sys.executable, '-B', str(work / script), *arguments],
                                     cwd=work, env=env, capture_output=True, encoding='utf-8', errors='replace', timeout=900)
                print(f'{"PASS" if run.returncode == 0 else "FAIL"} {label}', flush=True)
                if run.returncode:
                    failures.append(label)
                # Keep useful numbers and mutation evidence, without model-loading progress bars.
                output = run.stdout + run.stderr
                for line in output.splitlines():
                    if line.startswith(('PASS', 'FAIL', 'ERROR', 'METRIC', 'calibration', 'Historical')):
                        print('  ' + line, flush=True)
                if run.returncode:
                    print(output[-4000:], flush=True)
            except (OSError, subprocess.TimeoutExpired) as exc:
                failures.append(label)
                print(f'FAIL {label}: {exc}', flush=True)
        try:
            run = subprocess.run([sys.executable, '-B', str(work / 'tools/smoke_test.py'), '--quick', '--published'],
                                 cwd=work, env=env, timeout=1200)
            print(f'{"PASS" if run.returncode == 0 else "FAIL"} fresh-clone smoke --quick', flush=True)
            if run.returncode:
                failures.append('smoke --quick')
        except (OSError, subprocess.TimeoutExpired) as exc:
            failures.append('smoke --quick')
            print(f'FAIL smoke --quick: {exc}', flush=True)
    print(f'{"FAIL" if failures else "PASS"} check_all: {len(CHECKS) + len(REPO_CHECKS) + 1} test groups; {len(failures)} failures', flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())


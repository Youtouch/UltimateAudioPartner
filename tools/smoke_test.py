"""Exercise a fresh public copy and a synthetic sound library; no studio audio.

python tools/smoke_test.py [--quick] [--cpu] [--published] [--report result.json]
Default: use the installed live skill if present, as the maintainer's sync does.
--published tests exactly the skill shipped in this repository.
All model calls are offline: cache the models first for a fresh installation.
"""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import soundfile as sf

from sync_from_live import copy_repo_files, copy_skill


def snapshot(repo, dst, skill):
    """Tracked working files, plus the new public tool/template sources awaiting review."""
    dst.mkdir(parents=True)
    if (repo / '.git').exists():
        run = subprocess.run(['git', 'ls-files', '-z'], cwd=repo, capture_output=True, check=True)
        for raw in run.stdout.split(b'\0'):
            if not raw:
                continue
            rel = Path(os.fsdecode(raw))
            if rel.parts[0] == 'skills' or rel.as_posix() == 'tools/private_words.txt':
                continue
            source = repo / rel
            if source.is_file():
                (dst / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dst / rel)
    copy_repo_files(repo, dst)
    copy_skill(skill, dst / 'skills/audio-designer')


def command(args, cwd, env, timeout=900):
    run = subprocess.run([sys.executable, '-B', *map(str, args)], cwd=cwd, env=env,
                         capture_output=True, encoding='utf-8', errors='replace', timeout=timeout)
    if run.returncode:
        raise AssertionError(f'exit {run.returncode}: {" ".join(map(str, args))}\n' +
                             (run.stdout + '\n' + run.stderr)[-5000:])
    return run


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def audio_library(root, count, scripts):
    sys.path.insert(0, str(scripts))
    import audio_map
    sr = 48000
    rng = np.random.default_rng(1842)

    def tone(seconds, hz=330, amplitude=.25):
        return amplitude * np.sin(2 * np.pi * hz * np.arange(round(seconds * sr)) / sr)

    def rain(seconds):
        # A simple low-pass filter shapes deterministic broadband noise.
        from scipy.signal import lfilter
        return lfilter([.12], [1, -.88], rng.normal(0, .17, round(seconds * sr)))

    t = np.arange(10 * sr) / sr
    voice = tone(10, 220, .2) * (((t >= 1) & (t < 3)) | ((t >= 8) & (t < 10)))
    stereo_voice = np.column_stack((voice, voice))
    voice *= 10 ** ((-16 - audio_map.loudness_samples(stereo_voice, sr)['integrated_lufs']) / 20)
    rain5 = rain(5)
    click = np.zeros(round(.2 * sr))
    click[2400:2500] = .6 * np.hanning(100)
    vt = np.arange(5 * sr) / sr
    vibrato = .15 * np.sin(2 * np.pi * 220 * vt + 2 * np.sin(2 * np.pi * 5 * vt))
    burst = rng.normal(0, .08, 5 * sr) * (np.arange(5 * sr) % sr < sr // 5)
    specs = [
        ('Weather Pack/nested/rain approved é.wav', rain5, False),
        ('Weather Pack/nested/rain near duplicate 日本.wav', rain5 * .999, False),
        ('Effects Pack/click with spaces.wav', click, True),
        ('Music Pack/vibrato sung-like.wav', vibrato, False),
        ('Music Pack/nested/long tone 90.wav', tone(90), True),
        ('Weather Pack/nested/long rain 90.wav', rain(90), False),
        ('Effects Pack/noise burst.wav', burst, True),
        ('Music Pack/short tone é 日本.mp3', tone(.2, 660), False),
        ('Voice Pack/voice café.wav', voice, True),
    ]
    for i in range(count - 10):
        duration = (.2, 5, 90)[i % 3]
        samples = tone(duration, 120 + 27 * i, .1) if i % 2 else rain(duration)
        specs.append((f'Extra Pack/nest {i % 4}/synthetic {i:02d}.wav', samples, bool(i % 2)))
    expected = {}
    for name, y, stereo in specs:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if stereo:
            y = np.column_stack((y, y * .98))
        if path.suffix == '.mp3':
            intermediate = path.with_suffix('.source.wav')
            sf.write(intermediate, y, sr, subtype='PCM_16')
            subprocess.run([audio_map.media_tool(), '-v', 'error', '-threads', '1', '-i',
                            str(intermediate), '-codec:a', 'libmp3lame', str(path)], check=True)
            intermediate.unlink()
        else:
            sf.write(path, y, sr, subtype='PCM_16')
        expected[name] = len(y) / sr
    corrupt = 'Broken Pack/deliberately corrupt.wav'
    (root / corrupt).parent.mkdir(parents=True)
    (root / corrupt).write_bytes(b'This deliberately is not an audio file.\n')
    return expected, corrupt


def http(base, endpoint, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + endpoint, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = response.read()
        return payload if endpoint == '/' else json.loads(payload)


def exercise(copy, root, expected, corrupt, env, results):
    skill = copy / 'skills/audio-designer'
    scripts = skill / 'scripts'
    timings = {}

    def step(name, action):
        began = time.monotonic()
        try:
            action()
        except Exception as exc:
            print(f'FAIL {name}: {exc}', flush=True)
            results.append({'step': name, 'pass': False, 'error': str(exc)})
        else:
            print(f'PASS {name}', flush=True)
            results.append({'step': name, 'pass': True})
        timings[name] = time.monotonic() - began

    def tag():
        start = time.monotonic()
        run = command([scripts / 'audio_tag.py', 'tag', '--root', root, '--out', root / 'AUDIO_TAGS.csv'], copy, env)
        timings['tag_seconds'] = time.monotonic() - start
        print(f'INFO tagging {len(expected)} readable files: {timings["tag_seconds"]:.2f} s '
              f'({timings["tag_seconds"] / len(expected):.3f} s/file, startup included)', flush=True)
        rows = {r['relative_path']: r for r in read_csv(root / 'AUDIO_TAGS.csv')}
        assert set(rows) == set(expected)|{corrupt}, f'rows {len(rows)}, expected readable files plus explicit corrupt row'
        assert rows[corrupt]['recognition_status']=='unreadable' and rows[corrupt]['clap_row']=='-1' and not rows[corrupt]['top_labels']
        assert all(r.get('clap_row') and r.get('tagger') for p,r in rows.items() if p!=corrupt), 'missing fingerprints/tagger'
        skipped = read_csv(root / 'AUDIO_TAGS_skipped.csv')
        assert len(skipped) == 1 and skipped[0]['relative_path'] == corrupt, skipped
        assert skipped[0].get('reason'), 'corrupt file has no skip reason'
        output = run.stdout + run.stderr
        assert all(word in output.lower() for word in ('tagged', 'unchanged', 'skipped')), output
        ledger = {r['relative_path']: r for r in read_csv(root / 'AUDIO_LEDGER_files.csv')}
        assert set(expected) <= ledger.keys(), 'readable files missing from ledger'
        for path, seconds in expected.items():
            assert abs(float(ledger[path]['duration_seconds']) - seconds) < .1, (path, ledger[path])
            assert ledger[path]['pack'] == path.split('/')[0], ledger[path]
        resume = command([scripts / 'audio_tag.py', 'tag', '--root', root, '--out', root / 'AUDIO_TAGS.csv'], copy, env)
        assert len(read_csv(root / 'AUDIO_TAGS.csv')) == len(expected)+1, 'resume duplicated unchanged tags'
        assert corrupt in (resume.stdout + resume.stderr), 'corrupt file lost on resumed pass'

    step('tag + complete ledger + corrupt summary + resume', tag)
    step('names (launcher command)', lambda: command([scripts / 'audio_search.py', 'names', '--root', root], copy, env))

    def describe():
        start = time.monotonic()
        command([scripts / 'audio_describe.py', 'describe', '--root', root], copy, env)
        timings['describe_seconds'] = time.monotonic() - start
        print(f'INFO describing {len(expected)} readable files: {timings["describe_seconds"]:.2f} s '
              f'({timings["describe_seconds"] / len(expected):.3f} s/file, startup included)', flush=True)
        rows = {r['relative_path']: r for r in read_csv(root / 'AUDIO_DESCRIPTORS.csv')}
        assert set(rows) == set(expected), f'described {len(rows)}, expected {len(expected)}'
        for path, seconds in expected.items():
            assert abs(float(rows[path]['duration_s']) - seconds) < .1, rows[path]

    step('describe: measured descriptors for every readable file (launcher command)', describe)
    step('vocab_tags build (launcher command)', lambda: command([skill / 'tagging/vocab_tags.py', 'build', '--root', root], copy, env))
    step('sfx_groups build (launcher command)', lambda: command([scripts / 'sfx_groups.py', 'build', '--root', root], copy, env))

    def find():
        data = json.loads(command([scripts / 'audio_search.py', 'find', '--root', root, 'rain', '--top', '20'], copy, env).stdout)
        assert data['results'] and any('rain' in r['path'] for r in data['results']), data
        assert all(r['evidence'] for r in data['results']), 'search lost per-source evidence'
    step('find by words with evidence', find)

    def like():
        data = json.loads(command([scripts / 'audio_search.py', 'like', '--root', root,
                                  '--approved', 'rain approved', '--min-seconds', '60', '--top', '100'], copy, env).stdout)
        assert data, 'duration-filtered like returned nothing'
        assert all(abs(expected[r['path']] - 90) < .1 and r['seconds'] >= 60 for r in data), data
    step('like --min-seconds 60 returns only 90-second files', like)

    log = root.parent / 'browser.log'
    with socket.socket() as free:
        free.bind(('127.0.0.1', 0))
        port = free.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    with log.open('w', encoding='utf-8') as output:
        server = subprocess.Popen([sys.executable, '-B', str(scripts / 'sound_browser.py'),
                                   '--root', str(root), '--port', str(port), '--no-open'],
                                  cwd=copy, env=env, stdout=output, stderr=subprocess.STDOUT)
        try:
            ready = False
            until = time.monotonic() + 30
            while time.monotonic() < until and server.poll() is None:
                try:
                    with urllib.request.urlopen(base + '/', timeout=1) as response:
                        ready = response.status == 200
                except (OSError, urllib.error.URLError):
                    pass
                if ready:
                    break
                time.sleep(.1)
            if not ready:
                raise AssertionError('browser did not start: ' + log.read_text(encoding='utf-8'))

            def data_check():
                page = http(base, '/')
                assert b'Sound Browser' in page
                data = http(base, '/api/data')
                rows = {row[0]: row for row in data['rows']}
                assert set(expected) <= rows.keys(), 'browser omits readable files'
                assert corrupt in rows, 'browser omits corrupt ledger entry'
                for path, seconds in expected.items():
                    row = rows[path]
                    assert row[1] == path.split('/')[0] and row[3] is not None and abs(row[3] - seconds) < .1, row
            step('browser GET / and /api/data: all files, durations, packs', data_check)
            step('browser /api/search?q=rain', lambda: (
                assert_scores(http(base, '/api/search?q=rain'))))
            rows = {r['relative_path']: r for r in read_csv(root / 'AUDIO_TAGS.csv')}
            approved = 'Weather Pack/nested/rain approved é.wav'
            duplicate = 'Weather Pack/nested/rain near duplicate 日本.wav'
            step('browser audio-only /api/similar', lambda: assert_scores(
                http(base, '/api/similar?row=' + rows[approved]['clap_row'])))

            def verdict():
                http(base, '/api/verdict', {'path': approved, 'verdict': 'like', 'note': 'synthetic example'})
                http(base, '/api/verdict', {'path': 'Music Pack/nested/long tone 90.wav', 'verdict': 'reject'})
                ranked = http(base, '/api/approved?avoid=')
                assert ranked['results'] and ranked['results'][0]['path'] == duplicate, ranked
                assert all(r['path'] not in ranked['approved'] + ranked['rejected'] for r in ranked['results']), ranked
                avoided = http(base, '/api/approved?avoid=' + urllib.parse.quote('near duplicate'))
                assert all(r['path'] != duplicate for r in avoided['results']), avoided
                one = http(base, '/api/like?p=' + urllib.parse.quote(approved) + '&avoid=')
                assert one['results'][0]['path'] == duplicate, one
            step('browser approved/rejected ranking, avoid and per-file more-like', verdict)
        except Exception as exc:
            step('browser startup', lambda: (_ for _ in ()).throw(exc))
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

    def render():
        cue = root.parent / 'synthetic cues.csv'
        columns = ['layer', 'file', 'excerpt_start', 'excerpt_end', 'cue_in', 'cue_out', 'gain_db', 'duck_ratio', 'fade_in', 'fade_out']
        with cue.open('w', encoding='utf-8', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(columns)
            writer.writerows([
                ['voice', str(root / 'Voice Pack/voice café.wav'), 0, 10, 0, 12, 0, '-', 0, 0],
                ['music', str(root / 'Music Pack/nested/long tone 90.wav'), 0, 12, 0, 12, -12, 3, .01, .02],
                ['bed', str(root / 'Weather Pack/nested/long rain 90.wav'), 0, 12, 0, 12, -30, 8, .01, .02],
                ['sfx', str(root / 'Effects Pack/click with spaces.wav'), 0, .2, 5, 5.2, -15, '-', .005, .02],
            ])
        out = root.parent / 'rendered'
        command([skill / 'render/render_mix.py', 'render', cue, '--out', out], copy, env)
        report = (out / 'cue_report.md').read_text(encoding='utf-8')
        assert not re.search(r'\[FAIL\]|\|\s*FAIL\s*\|', report), report
        assert (out / 'final.wav').stat().st_size > 1000
        command([skill / 'mix/mix_check.py', 'stems', '--voice', out / 'voice.wav', '--music', out / 'music.wav',
                 '--bed', out / 'bed.wav', '--sfx', out / 'sfx.wav', '--mix', out / 'final.wav'], copy, env)
    step('cue render exits zero + report has no FAIL + mix_check', render)
    return timings


def assert_scores(data):
    assert data['scores'] and any(s is not None for s in data['scores']), data


def main():
    parser = argparse.ArgumentParser(description='Exercise a fresh public copy with a generated synthetic audio library.',
        epilog='Example: python tools/smoke_test.py --quick --published',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='10 files rather than 40 (one is corrupt)')
    parser.add_argument('--cpu', action='store_true', help='hide GPUs from torch, include wall time per readable file')
    parser.add_argument('--published', action='store_true', help='use only this copy, never an installed skill')
    parser.add_argument('--from', dest='source', type=Path, help='explicit live skill source')
    parser.add_argument('--report', type=Path, help='optional JSON test and timing report path')
    parser.add_argument('--keep', action='store_true', help='keep the synthetic workspace for diagnosis')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    live = Path.home() / '.agents/skills/audio-designer'
    source = args.source or (live if not args.published and live.is_dir() else repo / 'skills/audio-designer')
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    if args.cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    root = Path(tempfile.mkdtemp(prefix='uap-smoke-'))
    results, timings = [], {}
    try:
        copy = root / 'fresh clone'
        snapshot(repo, copy, source)
        print('PASS fresh public snapshot (allowlisted skill, no private library)', flush=True)
        library = root / 'synthetic sounds é 日本'
        library.mkdir()
        expected, corrupt = audio_library(library, 10 if args.quick else 40, copy / 'skills/audio-designer/scripts')
        print(f'PASS generated {len(expected)} readable files + one corrupt file', flush=True)
        timings = exercise(copy, library, expected, corrupt, env, results)
    except Exception as exc:
        print(f'FAIL setup: {exc}', flush=True)
        results.append({'step': 'setup', 'pass': False, 'error': str(exc)})
    finally:
        data = {'passed': all(r['pass'] for r in results) and bool(results), 'cpu': args.cpu,
                'files': 10 if args.quick else 40, 'steps': results, 'timings': timings}
        if args.report:
            args.report.write_text(json.dumps(data, indent=2), encoding='utf-8')
        if args.keep:
            print(f'INFO synthetic workspace retained: {root}', flush=True)
        else:
            shutil.rmtree(root)
    print(f'{"PASS" if data["passed"] else "FAIL"} smoke test ({sum(r["pass"] for r in results)}/{len(results)} steps)', flush=True)
    return 0 if data['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())


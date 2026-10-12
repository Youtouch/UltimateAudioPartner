"""Build an offline synthetic sound library and open the local browser."""
import argparse
import contextlib
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

from smoke_test import audio_library

REPO = Path(__file__).resolve().parents[1]


def skill_source(repo=REPO):
    """Use this published skill; a maintainer checkout can use its installed live source."""
    published = Path(repo) / 'skills/audio-designer'
    live = Path.home() / '.agents/skills/audio-designer'
    # The new helper distinguishes a complete published copy from a maintainer's older mirror.
    if (published / 'scripts/demo_metadata.py').is_file():
        return published
    if (live / 'scripts/demo_metadata.py').is_file():
        return live
    raise FileNotFoundError('Demo helper missing. Use a complete UAP release or sync the live skill.')


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def game_sounds(root, sr=48000):
    """Four deliberately designed signals; their names describe the generator, not recognition."""
    rng = np.random.default_rng(926)
    def clock(seconds):
        return np.arange(round(seconds * sr)) / sr
    t = clock(.55)
    coin = .32 * (np.sin(2 * np.pi * 1480 * t) + .4 * np.sin(2 * np.pi * 2220 * t))
    coin *= np.minimum(t / .003, 1) * np.exp(-t / .11)
    t = clock(.65)
    frequency = 1900 * np.exp(-t / .2) + 90
    laser = .5 * np.sin(2 * np.pi * np.cumsum(frequency) / sr)
    laser *= np.minimum(t / .004, 1) * np.exp(-t / .14)
    t = clock(.32)
    footstep = .6 * np.sin(2 * np.pi * 75 * t) * np.exp(-t / .045)
    footstep += .13 * rng.standard_normal(len(t)) * np.exp(-t / .018)
    footstep *= np.minimum(t / .002, 1)
    t = clock(.12)
    click = (.24 * rng.standard_normal(len(t)) + .14 * np.sin(2 * np.pi * 3400 * t))
    click *= np.minimum(t / .001, 1) * np.exp(-t / .006)
    sounds = {'Game SFX/Coin/coin_chirp_01.wav': coin, 'Game SFX/Laser/laser_sweep_01.wav': laser,
              'Game SFX/Footsteps/footstep_thud_01.wav': footstep, 'Game SFX/UI/ui_click_01.wav': click}
    for name, samples in sounds.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, samples, sr, subtype='PCM_16')


def build_demo(out=None, source=None, workers=1):
    """Build a new scratch library; refuse a populated destination so originals stay untouched."""
    skill = Path(source) if source else skill_source()
    scripts = skill / 'scripts'
    sys.path.insert(0, str(scripts))
    import audio_describe
    import audio_map
    import demo_metadata
    import library_metadata
    import sfx_groups
    root = Path(out).resolve() if out else Path(tempfile.mkdtemp(prefix='uap-demo-'))
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Demo destination must be empty: {root}')
    root.mkdir(parents=True, exist_ok=True)
    print('Building synthetic demo: no recognition models or downloads.', file=sys.stderr, flush=True)
    _, corrupt = audio_library(root, 10, scripts)
    (root / corrupt).unlink()
    (root / corrupt).parent.rmdir()
    # The smoke test's long-file cases exercise duration handling. Keep the demo fast to prepare and audition.
    for path in root.rglob('*.wav'):
        info = sf.info(path)
        if info.duration > 10:
            samples, sr = sf.read(path, frames=5 * info.samplerate, always_2d=True)
            path.unlink()
            sf.write(path.with_name(path.name.replace('90', '05')), samples, sr, subtype='PCM_16')
    game_sounds(root)
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in audio_describe.AUDIO)
    library_metadata.refresh_ledger(root, paths)
    ledger = []
    with (root / 'AUDIO_LEDGER_files.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            row['bucket'] = ('game_sfx' if row['relative_path'].startswith('Game SFX/') else
                             'ambience' if row['relative_path'].startswith('Weather Pack/') else
                             'music' if row['relative_path'].startswith('Music Pack/') else 'effects')
            ledger.append(row)
    write_csv(root / 'AUDIO_LEDGER_files.csv', list(ledger[0]), ledger)
    fields = ['relative_path', 'size_bytes', 'mtime_ns', 'seconds_tagged', 'speech', 'music',
              'silence', 'top_labels', 'timeline', 'clap_row', 'tagger']
    rows, vectors, vocab = [], [], []
    categories = [('coin', 'coin'), ('laser', 'laser'), ('footstep', 'footsteps'), ('click', 'click'),
                  ('rain', 'rain'), ('tone', 'tonal'), ('vibrato', 'tonal'), ('voice', 'unknown_source'),
                  ('noise', 'noise')]
    for number, path in enumerate(paths):
        rel, stat = path.relative_to(root).as_posix(), path.stat()
        samples, sr = audio_map.load_audio(path, seconds=2, sample_rate=16000, mono=True)
        vectors.append(demo_metadata.spectrum_vector(samples, sr))
        rows.append(dict(relative_path=rel, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                         seconds_tagged=sf.info(path).duration, speech=0, music=0, silence=0,
                         top_labels='', timeline='', clap_row=number, tagger='synthetic-generator-v1'))
        for needle, tag in categories:
            if needle in rel.lower():
                vocab.append(dict(relative_path=rel, tag=tag, name='1', label='', audio='',
                                  sources='generator', tagger='synthetic-generator-v1', built_at=''))
    write_csv(root / 'AUDIO_TAGS.csv', fields, rows)
    np.stack(vectors).astype('<f4').tofile(root / 'AUDIO_TAGS_clap.f32')
    write_csv(root / 'AUDIO_VOCAB_TAGS.csv', ['relative_path', 'tag', 'name', 'label', 'audio', 'sources', 'tagger', 'built_at'], vocab)
    marker = dict(schema=demo_metadata.SCHEMA,
                  labels='Synthetic generator tags: single source, not recognition or independent agreement.',
                  search='Demo search uses file names and measured words; semantic model search is available on a normally tagged library.',
                  similarity='Demo similarity uses measured spectrum, not CLAP fingerprints.',
                  files=len(paths))
    (root / 'UAP_DEMO.json').write_text(json.dumps(marker, indent=2) + '\n', encoding='utf-8')
    (root / 'DEMO_README.md').write_text(
        '# Synthetic UAP demo\n\nAll audio was generated locally for this demo. No recordings or downloaded models.\n\n'
        'Descriptors are measured from the real samples. Category tags come from the generator and count as one '
        'source; they do not claim model recognition or independent agreement. The demo searches names and measured '
        'words. Similarity uses a simple measured spectrum; a normally tagged library uses CLAP instead.\n\n'
        'This scratch folder is safe to delete after closing the browser. Export to a different folder.\n', encoding='utf-8')
    audio_describe.library(root, workers=workers)
    with contextlib.redirect_stdout(sys.stderr):
        sfx_groups.build(root)
    print(f'Demo library ready: {root}', file=sys.stderr, flush=True)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Example: python tools/demo.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out', type=Path, help='empty scratch library folder; otherwise create a temporary folder')
    parser.add_argument('--build-only', action='store_true', help='prepare the demo and print its path without starting the browser')
    parser.add_argument('--no-open', action='store_true', help='start the browser server without opening a browser tab')
    parser.add_argument('--port', type=int, default=8790, help='first local browser port to try')
    parser.add_argument('--workers', type=int, default=1, help='descriptor workers; one keeps a first demo light')
    parser.add_argument('--json', action='store_true', help='print the library path as JSON for scripts')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    # Even optional dependencies must never reach a model download during this command.
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    try:
        source = skill_source()
        root = build_demo(args.out, source, args.workers)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({'root': str(root)}) if args.json else str(root), flush=True)
    if args.build_only:
        return 0
    command = [sys.executable, str(source / 'scripts/sound_browser.py'), '--root', str(root), '--port', str(args.port)]
    if args.no_open:
        command.append('--no-open')
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())

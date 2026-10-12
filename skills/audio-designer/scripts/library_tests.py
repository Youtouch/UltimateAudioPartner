"""Temporary-library regressions for tagging, metadata readers and media discovery; no models."""
import contextlib
import csv
import io
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
if not (HERE / 'audio_tag.py').exists():
    HERE = Path(os.environ['UAP_LIVE']) / 'scripts'
sys.path.insert(0, str(HERE))
import audio_map
import audio_tag
import audio_index
import audio_search
try:
    import library_metadata
except ImportError:
    library_metadata = None  # The first regression run also runs against the version predating the helper.


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as target:
        writer = csv.DictWriter(target, fields)
        writer.writeheader()
        writer.writerows(rows)


def fake_decode(path, seconds):
    # Real container decoding catches the deliberately corrupt file; recognition is stubbed.
    y, sr = sf.read(path, always_2d=True, dtype='float32')
    n = min(len(y), int(seconds * sr))
    return np.zeros(n * 48000 // sr, 'float32'), np.zeros(n * 16000 // sr, 'float32')


def run_tag(root):
    capture = io.StringIO()
    result = {'speech': 0, 'music': 0, 'silence': 0, 'labels': {'Sine wave': .9}, 'timeline': []}
    with contextlib.redirect_stderr(capture), patch.object(audio_tag, 'decode', side_effect=fake_decode), \
            patch.object(audio_tag, 'tag_samples', return_value=(None, None, None)), \
            patch.object(audio_tag, 'summarise', return_value=result), \
            patch.object(audio_tag, 'embed_audio', return_value=np.eye(1, 512, dtype='float32')):
        audio_tag.library(root, None, root / 'AUDIO_TAGS.csv', 60)
    return capture.getvalue()


def tagging_contract():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pack = root / 'Pack espace é 日本'
        pack.mkdir()
        original = pack / 'rain.wav'
        sf.write(original, np.zeros((16000, 2)), 8000)
        corrupt = pack / 'corrupt.wav'
        corrupt.write_bytes(b'not an audio container')
        ledger = root / 'AUDIO_LEDGER_files.csv'
        fields = ['relative_path', 'pack', 'duration_seconds', 'sample_rate', 'channels',
                  'size_bytes', 'mtime_ns', 'licence', 'studio_note']
        rel = original.relative_to(root).as_posix()
        write_csv(ledger, fields, [
            {'relative_path': rel, 'pack': 'custom pack', 'duration_seconds': 999,
             'sample_rate': 99, 'channels': 9, 'size_bytes': 1, 'mtime_ns': 1,
             'licence': 'keep licence', 'studio_note': 'keep exact note'},
            {'relative_path': 'old/missing.wav', 'licence': 'retain missing row', 'studio_note': 'history'},
        ])
        output = run_tag(root)
        columns, rows = read_csv(ledger)
        by_path = {r['relative_path']: r for r in rows}
        assert columns == fields, columns
        assert len(rows) == 3 and set(by_path) == {rel, 'old/missing.wav', corrupt.relative_to(root).as_posix()}, rows
        updated = by_path[rel]
        assert updated['licence'] == 'keep licence' and updated['studio_note'] == 'keep exact note', updated
        assert float(updated['duration_seconds']) == 2 and int(updated['sample_rate']) == 8000 and int(updated['channels']) == 2, updated
        assert updated['size_bytes'] == str(original.stat().st_size) and updated['mtime_ns'] == str(original.stat().st_mtime_ns), updated
        assert by_path['old/missing.wav']['licence'] == 'retain missing row', by_path
        _, skipped = read_csv(root / 'AUDIO_TAGS_skipped.csv')
        assert len(skipped) == 1 and skipped[0]['relative_path'].endswith('corrupt.wav') and skipped[0]['reason'], skipped
        assert 'tagged=1 unchanged=0 skipped=1' in output, output
        # An unchanged pass does not probe again. Corrupt entries without duration are retried.
        real_measure = library_metadata.measure_metadata
        def only_retry_corrupt(path):
            assert path == corrupt, 'unchanged file re-probed'
            return real_measure(path)
        with patch.object(library_metadata, 'measure_metadata', side_effect=only_retry_corrupt):
            output = run_tag(root)
        assert 'tagged=0 unchanged=1 skipped=1' in output, output
        sf.write(original, np.zeros((24000, 1)), 8000)
        sf.write(pack / 'new.wav', np.zeros((4000, 1)), 8000)
        output = run_tag(root)
        _, rows = read_csv(ledger)
        by_path = {r['relative_path']: r for r in rows}
        assert len(rows) == 4 and float(by_path[rel]['duration_seconds']) == 3, rows
        assert int(by_path[rel]['channels']) == 1 and by_path[rel]['licence'] == 'keep licence', by_path[rel]
        assert 'tagged=2 unchanged=0 skipped=1' in output, output
        corrupt.unlink()
        output = run_tag(root)
        _, skipped = read_csv(root / 'AUDIO_TAGS_skipped.csv')
        assert not skipped and 'tagged=0 unchanged=2 skipped=0' in output, (skipped, output)
        _, rows = read_csv(ledger)
        assert len(rows) == 4, 'Refreshing must not remove a disappeared row'
        ledger_before = ledger.read_bytes()
        with patch.object(library_metadata.os, 'replace', side_effect=OSError('simulated replace failure')):
            try:
                library_metadata.write_csv_atomic(ledger, columns, [])
            except OSError:
                pass
            else:
                raise AssertionError('Atomic replace failure was swallowed')
        assert ledger.read_bytes() == ledger_before and not list(root.glob(ledger.name + '.*.tmp'))
        # A long file retains its full duration even when recognition is capped at the first minute.
        long_file = pack / 'long.wav'
        sf.write(long_file, np.zeros((90 * 8000, 1)), 8000)
        run_tag(root)
        _, rows = read_csv(ledger)
        assert float(next(row for row in rows if row['relative_path'].endswith('/long.wav'))['duration_seconds']) == 90
        print('PASS library tagging: preserve all columns/rows, refresh new/changed metadata, skipped reasons and summary')


def no_ledger_contract():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        tags = [dict(relative_path='Pack/rain.wav', seconds_tagged='90', top_labels='Rain 0.9', silence='0', clap_row='0'),
                dict(relative_path='Pack/click.wav', seconds_tagged='5', top_labels='Click 0.9', silence='0', clap_row='1'),
                dict(relative_path='Pack/music.wav', seconds_tagged='20', top_labels='Music 0.9', silence='0', clap_row='2')]
        write_csv(root / 'AUDIO_TAGS.csv', list(tags[0]), tags)
        np.eye(3, 512, dtype='<f4').tofile(root / 'AUDIO_TAGS_clap.f32')
        capture = io.StringIO()
        with contextlib.redirect_stderr(capture), patch.object(audio_search.Index, '_name_vectors', return_value=np.zeros((3, 512))):
            index = audio_search.Index(root)
            audio_search.Index(root)
        assert np.array_equal(index.seconds, [90, 5, 20]), index.seconds
        assert capture.getvalue().count('NOTE') == 1 and 'AUDIO_LEDGER_files.csv' in capture.getvalue(), capture.getvalue()
        assert audio_search.like(index, ['click'], min_seconds=60)[0]['path'] == 'Pack/rain.wav'
        library_metadata._missing_ledger_noted.clear()
        capture = io.StringIO()
        with contextlib.redirect_stderr(capture):
            assert {p.name for p in audio_index.selected(root, None)} == {'rain.wav', 'click.wav'}, 'tag classifications must exclude positively tagged music'
            audio_index.selected(root, None)
        assert capture.getvalue().count('NOTE') == 1, capture.getvalue()
        sys.path.insert(0, str(HERE.parent / 'tagging'))
        import vocab_tags
        library_metadata._missing_ledger_noted.clear()
        capture = io.StringIO()
        with contextlib.redirect_stderr(capture):
            loaded = vocab_tags.load_library(root)
            vocab_tags.load_library(root)
        assert capture.getvalue().count('NOTE') == 1, capture.getvalue()
        assert loaded.paths == ['Pack/click.wav', 'Pack/music.wav', 'Pack/rain.wav'] and loaded.packs == ['Pack'] * 3
        assert audio_search.clean_name('rain.wav') == 'rain'
        assert audio_search.clean_name('Pack\\weather\\Rain.wav', 'Pack') == 'weather rain'
        print('PASS library readers: missing-ledger note once, tag duration fallback, packs, non-music selection')


def media_tool_contract():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        binary = root / 'ffmpeg'
        sibling = root / 'ffprobe'
        binary.touch()
        sibling.touch()
        audio_map.media_tool.cache_clear()
        with patch.object(audio_map, 'os', SimpleNamespace(name='posix', environ={'FFMPEG': str(binary)})):
            assert audio_map.media_tool() == str(binary)
            assert audio_map.media_tool('ffprobe') == str(sibling)
        # A leftover Windows installation must not override PATH on macOS/Linux.
        winget = root / 'Microsoft' / 'WinGet' / 'Packages' / 'Gyan.FFmpeg.test' / 'version' / 'bin'
        winget.mkdir(parents=True)
        (winget / 'ffmpeg.exe').touch()
        audio_map.media_tool.cache_clear()
        with patch.object(audio_map, 'os', SimpleNamespace(name='posix', environ={'LOCALAPPDATA': str(root)})), \
                patch.object(audio_map.shutil, 'which', side_effect=lambda name: '/usr/bin/' + name):
            assert audio_map.media_tool() == '/usr/bin/ffmpeg'
            assert audio_map.media_tool('ffprobe') == '/usr/bin/ffprobe'
        audio_map.media_tool.cache_clear()
        print('PASS media discovery: extensionless override siblings and PATH on non-Windows')


def exact_examples_contract():
    index = SimpleNamespace(
        paths=['Pack/approved.wav', 'OtherPack/approved.wav', 'Pack/candidate.wav'],
        audio=np.array([[1., 0.], [0., 1.], [1., 0.]]),
        seconds=np.ones(3), label_text=[''] * 3)
    exact = audio_search.like(index, ['Pack/approved.wav'], exact=True)
    assert exact[0]['path'] == 'Pack/candidate.wav' and exact[0]['score'] == 1, exact
    assert [row['path'] for row in exact] == ['Pack/candidate.wav', 'OtherPack/approved.wav'], exact
    substring = audio_search.like(index, ['Pack/approved.wav'])
    assert len(substring) == 1 and substring[0]['score'] == .5, substring
    print('PASS exact approved examples: saved paths cannot select a substring collision')


def old_ledger_contract():
    """A ledger written before mtime_ns, with a pack finer than the first folder: keep the pack, never re-probe."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / 'Vendor' / 'fine_pack' / 'tone.wav'
        path.parent.mkdir(parents=True)
        sf.write(path, np.zeros(4800, dtype='float32'), 48000)
        write_csv(root / 'AUDIO_LEDGER_files.csv',
                  ['relative_path', 'pack', 'size_bytes', 'duration_seconds', 'sample_rate', 'channels', 'kind_guess'],
                  [{'relative_path': 'Vendor/fine_pack/tone.wav', 'pack': 'fine_pack', 'size_bytes': path.stat().st_size,
                    'duration_seconds': '0.1', 'sample_rate': '48000', 'channels': '1', 'kind_guess': 'sfx'}])
        with patch.object(library_metadata, 'measure_metadata', side_effect=AssertionError('re-probed an unchanged file')):
            assert library_metadata.refresh_ledger(root, [path]) == {}
        row = read_csv(root / 'AUDIO_LEDGER_files.csv')[1][0]
        assert row['pack'] == 'fine_pack', row
        assert row['kind_guess'] == 'sfx' and row['mtime_ns'] == str(path.stat().st_mtime_ns), row
        path.write_bytes(path.read_bytes() + bytes(16))  # a changed file is re-measured, and still keeps its pack
        library_metadata.refresh_ledger(root, [path])
        row = read_csv(root / 'AUDIO_LEDGER_files.csv')[1][0]
        assert row['pack'] == 'fine_pack' and row['size_bytes'] == str(path.stat().st_size), row
    print('PASS old_ledger_contract')


def main():
    if not __debug__:
        raise RuntimeError('Self-tests require asserts: do not use python -O')
    failed = 0
    for test in (tagging_contract, no_ledger_contract, media_tool_contract, exact_examples_contract, old_ledger_contract):
        try:
            test()
        except Exception as exc:
            failed += 1
            print(f'FAIL {test.__name__}: {type(exc).__name__}: {exc}', file=sys.stderr)
    return int(bool(failed))


if __name__ == '__main__':
    sys.exit(main())

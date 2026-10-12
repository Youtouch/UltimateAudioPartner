"""Check cached waveform/spectrogram APIs and loudness matching on known synthetic audio.

Example: python scripts/visual_tests.py
"""
import sys
sys.dont_write_bytecode = True

import json
import os
from pathlib import Path
import tempfile
import threading
import struct
import zlib
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from unittest.mock import patch

import numpy as np
import soundfile as sf
import sound_browser


def read_rgb_png(body):
    """Validate the generated PNG's signature, chunks, CRCs and filter-0 RGB raster using only stdlib."""
    assert body[:8] == b'\x89PNG\r\n\x1a\n'
    cursor, compressed, shape = 8, bytearray(), None
    while cursor < len(body):
        size = struct.unpack('>I', body[cursor:cursor + 4])[0]
        kind = body[cursor + 4:cursor + 8]
        data = body[cursor + 8:cursor + 8 + size]
        crc = struct.unpack('>I', body[cursor + 8 + size:cursor + 12 + size])[0]
        assert zlib.crc32(kind + data) & 0xffffffff == crc
        if kind == b'IHDR':
            width, height, depth, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', data)
            assert (depth, color, compression, filtering, interlace) == (8, 2, 0, 0, 0)
            shape = (height, width)
        elif kind == b'IDAT':
            compressed.extend(data)
        cursor += 12 + size
        if kind == b'IEND':
            break
    assert cursor == len(body) and shape is not None
    height, width = shape
    rows = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(height, 1 + width * 3)
    assert np.all(rows[:, 0] == 0)
    return rows[:, 1:].reshape(height, width, 3)


def main():
    count, failed = 0, []

    def check(name, result):
        nonlocal count
        count += 1
        print(('PASS ' if result else 'FAIL ') + name)
        if not result:
            failed.append(name)

    with tempfile.TemporaryDirectory(prefix='uap-visual-') as tmp:
        root = Path(tmp)
        sr = 48000
        t = np.arange(sr * 4) / sr
        tone = np.sin(2 * np.pi * 997 * t)
        for name, data in [('quiet.wav', tone * .1), ('loud.wav', tone * .5), ('silent.wav', tone * 0),
                           ('event.wav', tone[:sr // 8] * .2)]:
            sf.write(root / name, data, sr, subtype='FLOAT')
        # Opposed stereo must remain visible and measurable.
        sf.write(root / 'stereo.wav', np.column_stack([tone * .1, -tone * .1]), sr, subtype='FLOAT')
        (root / 'AUDIO_TAGS.csv').write_text('relative_path,clap_row,top_labels,timeline,speech,music,silence,seconds_tagged\n'
                                           'quiet.wav,,Tone 1,,0,0,0,4\nloud.wav,,Tone 1,,0,0,0,4\n', encoding='utf-8')
        lib = sound_browser.Library(root)
        check('browser implements waveform, spectrogram and 2-4-way comparison',
              all(callable(getattr(lib, n, None)) for n in ('waveform', 'spectrogram', 'compare')))
        if failed:
            return 1
        import audio_visuals
        server = sound_browser.Server(('127.0.0.1', 0), sound_browser.handler(lib))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        base = 'http://127.0.0.1:' + str(server.server_address[1])

        def request(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            req = Request(base + path, data=data, headers={'Content-Type': 'application/json'} if data else {})
            try:
                with urlopen(req, timeout=30) as response:
                    raw = response.read()
                    return response.status, json.loads(raw) if response.headers.get_content_type() == 'application/json' else raw
            except HTTPError as exc:
                return exc.code, json.loads(exc.read())

        try:
            status, wave = request('/api/waveform?' + urlencode({'p': 'quiet.wav'}))
            check('waveform HTTP response has fixed bins, real duration and nonzero sample peak',
                  status == 200 and len(wave['peaks']) == 128 and abs(wave['duration_s'] - 4) < .001
                  and .099 < max(wave['peaks']) < .101 and abs(wave['sample_peak'] - .1) < .00001)
            cache = audio_visuals.cache_path(root, root / 'quiet.wav')
            check('waveform stored beside descriptors with size, nanosecond mtime and algorithm version',
                  cache.parent == root / 'AUDIO_VISUALS' and cache.is_file()
                  and all(k in json.loads(cache.read_text()) for k in ('size_bytes', 'mtime_ns', 'version')))
            stamp = cache.stat().st_mtime_ns
            with patch.object(audio_visuals, '_analyse', side_effect=AssertionError('cached file was decoded')):
                check('unchanged waveform uses its disk cache without decoding the source',
                      sound_browser.Library(root).waveform('quiet.wav')['peaks'] == wave['peaks'])
            # The first list may need to fill one missing cache; warm it and prove subsequent lists never decode.
            lib.waveform('loud.wav')
            with patch.object(audio_visuals, '_analyse', side_effect=AssertionError('list decoded unchanged file')):
                status, payload = request('/api/data')
            check('list carries cached row waveforms without source decoding', status == 200
                  and set(payload['waveforms']) == {'quiet.wav', 'loud.wav'} and cache.stat().st_mtime_ns == stamp)
            original = (root / 'quiet.wav').stat()
            sf.write(root / 'quiet.wav', tone * .2, sr, subtype='FLOAT')
            os.utime(root / 'quiet.wav', ns=(original.st_atime_ns, original.st_mtime_ns + 1000000))
            newer = lib.waveform('quiet.wav')
            check('same-size replacement invalidates on nanosecond mtime and updates peaks',
                  (root / 'quiet.wav').stat().st_size == original.st_size and .199 < max(newer['peaks']) < .201)
            with (root / 'quiet.wav').open('ab') as target:
                target.write(b'changed-size')
            os.utime(root / 'quiet.wav', ns=(original.st_atime_ns, original.st_mtime_ns + 1000000))
            with patch.object(audio_visuals, '_analyse', wraps=audio_visuals._analyse) as analyse:
                lib.waveform('quiet.wav')
            check('same-mtime size change also invalidates waveform', analyse.call_count == 1)
            record = json.loads(cache.read_text())
            record['version'] = 'obsolete'
            cache.write_text(json.dumps(record))
            with patch.object(audio_visuals, '_analyse', wraps=audio_visuals._analyse) as analyse:
                sound_browser.Library(root).waveform('quiet.wav')
            check('cache algorithm version invalidates stale envelopes', analyse.call_count == 1)
            with patch.dict(sys.modules, {'PIL': None, 'PIL.Image': None, 'PIL.ImageDraw': None}):
                status, png = request('/api/spectrogram?' + urlencode({'p': 'quiet.wav'}))
            check('spectrogram rendering needs no optional Pillow dependency', status == 200)
            if status != 200:
                return 1
            pixels = read_rgb_png(png)
            check('spectrogram endpoint returns a real bounded PNG with frequency energy', status == 200
                  and pixels.shape == (224, 720, 3) and pixels.std() > 3
                  and len(png) < 300000)
            # A 997 Hz tone occupies the expected logarithmic frequency band, not arbitrary pixels.
            row = int(np.argmax(pixels[22:202, 10:710].mean(axis=(1, 2))))
            freq = 40 * (min(sr / 2, 20000) / 40) ** ((179 - row) / 179)
            check('spectrogram ridge agrees with the known 997 Hz tone', abs(freq - 997) < 130)
            with patch.object(audio_visuals, '_spectrogram', side_effect=AssertionError('cached spectrogram decoded')):
                check('spectrogram is cached across browser instances', sound_browser.Library(root).spectrogram('quiet.wav').read_bytes() == png)
            status, compared = request('/api/compare', {'paths': ['quiet.wav', 'loud.wav']})
            items = compared.get('items', [])
            check('compare returns two measured sounds and side-by-side key numbers', status == 200 and len(items) == 2
                  and all('numbers' in item and 'waveform' in item and item['match_basis'] == 'short-term 3 s' for item in items))
            check('gain maths removes a known amplitude advantage without boosting or clipping', len(items) == 2
                  and abs((items[1]['gain_db'] - items[0]['gain_db']) - 20 * np.log10(.2 / .5)) < .001
                  and abs(.2 * items[0]['gain'] - .5 * items[1]['gain']) < .00001
                  and all(0 < item['gain'] <= 1 and item['waveform']['true_peak_dbtp'] + item['gain_db'] <= -1 + .001 for item in items))
            event = lib.compare(['quiet.wav', 'event.wav'])['items'][1]
            check('sounds under 3 s use an explicitly labelled K-weighted event fallback', event['match_basis'] == 'event K-weighted (under 3 s)'
                  and np.isfinite(event['match_lufs']))
            silence = lib.compare(['quiet.wav', 'silent.wav'])['items'][1]
            check('digital silence has no invented loudness and remains silent', silence['match_lufs'] is None and silence['gain'] == 1)
            import audio_describe
            boundary = np.zeros(sr * 4)
            boundary[65536 - 8:65536 + 8] = [
                .07253607480006208, .33046144572999225, .2245542789208127, .43686801699942934,
                -.8672087595982559, .27766286951782304, .07571258775983736, .632413823555139,
                .7902526426128335, -.8769183867524867, .59099093638171, -.44401699677110323,
                .22447438140228393, .47595304833286234, .6245932032368021, .7930852504092653]
            sf.write(root / 'boundary.wav', boundary, sr, subtype='FLOAT')
            whole_peak = audio_describe.true_peak(boundary[:, None])
            boundary_wave = lib.waveform('boundary.wav')
            check('streaming true peak preserves inter-sample overs across a decode block boundary',
                  abs(boundary_wave['true_peak_dbtp'] - whole_peak) < .001 and whole_peak > 2)
            boundary_item = lib.compare(['boundary.wav', 'quiet.wav'])['items'][0]
            check('comparison peak ceiling includes inter-sample overs at decode block boundaries',
                  whole_peak + boundary_item['gain_db'] <= -1 + .001)
            check('comparison accepts the full four-sound selection', len(lib.compare(['quiet.wav', 'loud.wav', 'event.wav', 'silent.wav'])['items']) == 4)
            stereo = lib.waveform('stereo.wav')
            check('opposed stereo channels do not cancel the envelope or loudness', max(stereo['peaks']) > .099
                  and stereo['match_lufs'] > wave['match_lufs'] + 2.99)
            for paths in ([], ['quiet.wav'], ['quiet.wav'] * 2, ['quiet.wav', 'loud.wav', 'event.wav', 'silent.wav', 'stereo.wav']):
                check('compare rejects invalid selection ' + str(len(paths)), request('/api/compare', {'paths': paths})[0] == 400)
            for endpoint in ('waveform', 'spectrogram'):
                check(endpoint + ' refuses path traversal', request('/api/' + endpoint + '?' + urlencode({'p': '../outside.wav'}))[0] == 404)
            check('compare refuses paths outside the library', request('/api/compare', {'paths': ['quiet.wav', '../outside.wav']})[0] == 404)
            sf.write(root / 'quiet.wav', .2 * np.sin(2 * np.pi * 2500 * t), sr, subtype='FLOAT')
            with patch.object(audio_visuals, '_spectrogram', wraps=audio_visuals._spectrogram) as draw:
                updated_png = lib.spectrogram('quiet.wav').read_bytes()
            check('changed audio invalidates its spectrogram as well as its waveform', draw.call_count == 1 and updated_png != png)
            import audio_describe
            described = root / 'described'
            described.mkdir()
            sf.write(described / 'event.wav', tone[:sr // 8] * .2, sr, subtype='FLOAT')
            (described / 'broken.wav').write_bytes(b'not an audio file')
            def measured(path):
                if path.name == 'broken.wav':
                    raise ValueError('synthetic corruption')
                return {'duration_s': .125}
            with patch.object(audio_describe, 'describe_file', side_effect=measured):
                result = audio_describe.library(described, workers=1)
                check('descriptor pass precomputes waveforms and preserves corrupt-file skips', result['described'] == 1
                      and result['skipped'] == 1 and audio_visuals.cache_path(described, described / 'event.wav').is_file())
                audio_visuals.cache_path(described, described / 'event.wav').unlink()
                result = audio_describe.library(described, workers=1)
                check('unchanged descriptor rows still gain a missing visual cache', result['unchanged'] == 1
                      and result['described'] == 0 and audio_visuals.cache_path(described, described / 'event.wav').is_file())
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
    print(('FAIL' if failed else 'PASS') + f' visual_tests: {count} checks; {len(failed)} failures')
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())

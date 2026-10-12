"""Browse a sound library by what it contains: tags, description search, "more like this", audition.

    python sound_browser.py --root D:/MySounds       # opens the browser
    python sound_browser.py --root D:/MySounds --port 8790 --no-open

Runs only while this window is open (Ctrl+C stops it) and listens on 127.0.0.1 only.
Reads AUDIO_LEDGER_files.csv, AUDIO_TAGS.csv and AUDIO_TAGS_clap.f32 from the root, and when present
AUDIO_DESCRIPTORS.csv (measured words), AUDIO_VOCAB_TAGS.csv (studio tags) and AUDIO_GROUPS.csv (duplicates,
variation sets); writes your likes, rejects and notes to AUDIO_VERDICTS.json there, where agents read them
too. Exports go only to a folder you choose outside the library.
"""

import argparse
import csv
import json
import mimetypes
import re
import os
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from library_metadata import read_metadata
import audio_describe
import audio_visuals
import demo_metadata

HERE = Path(__file__).resolve().parent
SHOWN = list(audio_describe.NUMBERS)  # the measures each row carries, in this order
# Studio tags that describe a character or a material, not a kind of sound: never offered as example searches.
NOT_A_SOUND = {'single_event', 'repeated_events', 'soft', 'sharp_attack', 'very_quiet', 'silence_or_room_tone', 'unknown_source',
               'noise', 'calm', 'music', 'instrumental', 'ambience_bed', 'wood', 'metal', 'glass', 'stone', 'dirt', 'gravel',
               'snow', 'grass', 'water', 'flesh', 'retro_8bit'}
PAIR = re.compile(r'(.+?) (\d+(?:\.\d+)?)(?:, |; |$)')
DIM = 512


def pairs(text):
    return [(m[0], float(m[1])) for m in PAIR.findall(text or '')]


class Library:
    def __init__(self, root):
        self.root = root.resolve()
        self.tags = root / 'AUDIO_TAGS.csv'
        self.vectors = root / 'AUDIO_TAGS_clap.f32'
        self.verdicts_path = root / 'AUDIO_VERDICTS.json'
        self.lock = threading.Lock()
        self.visual_lock = threading.RLock()

    def payload(self):
        """Ledger rows joined with the latest tag row per path. Untagged files are listed too."""
        tagged = {}
        if self.tags.exists():
            with self.tags.open(encoding='utf-8-sig', newline='') as f:
                for r in csv.DictReader(f):
                    tagged[r['relative_path']] = r
        ledger = read_metadata(self.root, tagged)
        # Studio tags cross-referenced by tagging/vocab_tags.py: per file, tag -> which sources support it.
        # Bits: 1 name, 2 label, 4 audio. A tag with two or more bits is agreed; one bit is a candidate.
        vocab, vocab_index, vocab_tags = [], {}, {}
        vpath = self.root / 'AUDIO_VOCAB_TAGS.csv'
        if vpath.exists():
            with vpath.open(encoding='utf-8', newline='') as f:
                for r in csv.DictReader(f):
                    tag = r['tag']
                    if tag not in vocab_index:
                        vocab_index[tag] = len(vocab)
                        vocab.append(tag)
                    mask = (r['name'] == '1') | (bool(r['label']) << 1) | (bool(r['audio']) << 2)
                    if mask:
                        vocab_tags.setdefault(r['relative_path'], []).extend([vocab_index[tag], mask])
        names, index, rows = [], {}, []
        paths = sorted(set(ledger) | set(tagged))
        descriptors = audio_describe.read_descriptors(self.root)
        table = audio_describe.table_for(paths, descriptors) if descriptors else None
        masks = [audio_describe.word_mask(w[0], table) for w in audio_describe.WORDS] if table is not None else []
        groups = {}
        gpath = self.root / 'AUDIO_GROUPS.csv'
        if gpath.exists():
            with gpath.open(encoding='utf-8', newline='') as f:
                groups = {r['relative_path']: r for r in csv.DictReader(f)}

        def ids(items):
            out = []
            for name, score in items:
                if name not in index:
                    index[name] = len(names)
                    names.append(name)
                out += [index[name], score]
            return out

        for n, rel in enumerate(paths):
            led, tag = ledger.get(rel, {}), tagged.get(rel)
            group = groups.get(rel)
            timeline = []
            if tag:
                for part in tag.get('timeline', '').split(' | '):
                    span, _, labels = part.partition(': ')
                    if span:
                        timeline.append([float(span.split('-')[0]), ids(pairs(labels))])
            rows.append([rel, led.get('pack') or (rel.split('/')[0] if '/' in rel else self.root.name), led.get('bucket') or '',
                         float(led.get('duration_seconds') or 0) or None,
                         float(tag['speech']) if tag else None, float(tag['music']) if tag else None,
                         ids(pairs(tag['top_labels'])) if tag else [], timeline,
                         int(tag['clap_row']) if tag and tag.get('clap_row') else -1,
                         vocab_tags.get(rel, []),
                         [k for k, m in enumerate(masks) if m[n]],
                         [None if not np.isfinite(table[m][n]) else round(float(table[m][n]), 4) for m in SHOWN]
                         if table is not None and rel in descriptors else None,
                         [group['duplicate_set'], group['duplicate_of'], group['variation_set'], int(group['variation_size'] or 0)]
                         if group else None])
        words = [{'word': w[0], 'aliases': list(w[1]), 'explain': w[3], 'measures': [m for m, _, _ in w[2]],
                  'rule': ' and '.join(f'{m} {op} {v:g}' for m, op, v in w[2])} for w in audio_describe.WORDS]
        waveforms = {}
        for rel in paths:
            try:
                waveforms[rel] = self.waveform(rel)
            except (ValueError, OSError, subprocess.SubprocessError):
                pass  # corrupt/missing audio remains visible with its other evidence
        return {'labels': names, 'vocab': vocab, 'rows': rows, 'verdicts': self.verdicts(), 'root': str(self.root),
                'waveforms': waveforms, 'demo': demo_metadata.read_demo(self.root),
                'tagged': len(tagged), 'total': len(rows), 'words': words, 'measured': len(descriptors),
                'numbers': {m: list(audio_describe.NUMBERS[m]) + [audio_describe.MEASURE_EXPLAIN.get(m, '')] for m in SHOWN},
                'shown': SHOWN, 'examples': self.examples(rows, vocab, words)}

    @staticmethod
    def examples(rows, vocab, words):
        """Three example searches from this library's commonest agreed sound categories, each with the plain word
        its files most often earn: 'short click', 'loopable rain', ..."""
        from collections import Counter
        counts, earned = Counter(), {}
        for r in rows:
            v = r[9]
            for k in range(0, len(v), 2):
                tag = vocab[v[k]]
                if bin(v[k + 1]).count('1') >= 2 and tag not in NOT_A_SOUND:
                    counts[tag] += 1
                    earned.setdefault(tag, Counter()).update(
                        words[w]['word'] for w in r[10] if words[w]['word'] not in ('single', 'loud', 'quiet'))
        out = []
        for tag, _ in counts.most_common(3):
            word = earned[tag].most_common(1)
            out.append(((word[0][0] + ' ') if word else '') + tag.replace('_', ' '))
        return out

    def matrix(self):
        if not self.vectors.exists():
            return np.zeros((0, DIM), '<f4')
        data = np.fromfile(self.vectors, '<f4')
        return data[:len(data) // DIM * DIM].reshape(-1, DIM)  # a pass still writing may leave a partial row

    def similar(self, like):
        """More like this: the sound itself, by its audio fingerprint."""
        m = self.matrix()
        if not 0 <= like < len(m):
            raise ValueError('That file has no fingerprint yet')
        return {'scores': [round(float(s), 4) for s in m @ m[like]]}

    def index(self):
        """Cache the same search index used by the command-line tools."""
        from audio_search import Index
        stamp = self.tags.stat().st_mtime_ns
        with self.lock:
            stale = getattr(self, '_index_stamp', None) != stamp and time.time() - getattr(self, '_index_built', 0) > 600
            if getattr(self, '_index', None) is None or stale:
                self._index, self._index_stamp, self._index_built = Index(self.root), stamp, time.time()
            return self._index

    def ranked_payload(self, results):
        index = self.index()
        rows = dict(zip(index.paths, index.rows))
        scores = [None] * len(self.matrix())
        for result in results:
            row = int(rows[result['path']])
            if 0 <= row < len(scores):
                scores[row] = result['score']
        return {'scores': scores, 'results': results}

    def search(self, query):
        """Description search, with audio, name and label evidence kept separate."""
        results, wanted = self.index().find(query, top=1000)
        # The page sorts by score: give it the search's own order (files matching the plain words first).
        data = self.ranked_payload([dict(r, score=float(len(results) - k)) for k, r in enumerate(results)])
        data['matching'] = sum(1 for r in results if r.get('matches_words'))
        sound, filters = self.index().parse(query)
        data.update(evidence={r['row']: r['evidence'] for r in results}, labels=list(wanted), sound=sound, filters=filters)
        return data

    def more_like(self, path=None, avoid=''):
        """Learn from saved judgements, using the CLI's example ranking unchanged."""
        from audio_search import like
        index = self.index()
        verdicts = self.verdicts()
        rejected = [p for p, v in verdicts.items() if v.get('verdict') == 'reject' and p in index.paths]
        if path is not None:
            self.file(path)
            if path not in index.paths:
                raise ValueError('That file has no fingerprint yet')
            approved = [path]
        else:
            approved = [p for p, v in verdicts.items() if v.get('verdict') == 'like' and p in index.paths]
        if not approved:
            raise ValueError('Like a tagged sound first, then try More like my approved')
        if avoid:
            try:
                re.compile(avoid)
            except re.error as exc:
                raise ValueError(f'Invalid avoid expression: {exc}') from exc
        results = like(index, approved, rejected, avoid=avoid or None, top=1000, exact=True)
        data = self.ranked_payload(results)
        data.update(approved=approved, rejected=rejected)
        return data

    def varied(self, path, n):
        """The n most different takes in this file's variation set, for round-robin playback."""
        import sfx_groups
        self.file(path)
        picked = sfx_groups.varied(self.root, path, n)
        return self.ranked_payload([{'path': p, 'score': float(len(picked) - k)} for k, p in enumerate(picked)]) | {'picked': picked}

    def waveform(self, rel):
        with self.visual_lock:
            return audio_visuals.waveform(self.root, self.file(rel))

    def spectrogram(self, rel):
        with self.visual_lock:
            return audio_visuals.spectrogram(self.root, self.file(rel))

    def compare(self, paths):
        if not isinstance(paths, list) or not 2 <= len(paths) <= 4 or any(not isinstance(p, str) for p in paths):
            raise ValueError('Select two to four different sounds to compare')
        if len(set(paths)) != len(paths):
            raise ValueError('Select two to four different sounds to compare')
        # Resolve every source before preparing any caches.
        for path in paths:
            self.file(path)
        waves = [self.waveform(path) for path in paths]
        target, gains = audio_visuals.match_gains(waves)
        descriptors = audio_describe.read_descriptors(self.root)
        table = audio_describe.table_for(paths, descriptors)
        items = []
        for n, (path, wave, gain) in enumerate(zip(paths, waves, gains)):
            numbers = {m: None if not np.isfinite(table[m][n]) else round(float(table[m][n]), 4) for m in SHOWN}
            items.append({'path': path, 'waveform': wave, 'numbers': numbers,
                          'match_lufs': wave['match_lufs'], 'match_basis': wave['match_basis'], **gain})
        return {'items': items, 'target_lufs': round(target, 6), 'matched': True}

    def export(self, body):
        import export_sfx
        paths = body.get('paths') or []
        if not paths:
            raise ValueError('Select some sounds first')
        for p in paths:
            self.file(p)
        out = Path(str(body.get('out') or '').strip().strip('"'))
        if not str(out).strip() or not out.is_absolute():
            raise ValueError('Give the full path of the folder to export to')
        rows = export_sfx.export(self.root, paths, out, fmt=body.get('format', 'wav16'), rate=int(body.get('rate') or 48000),
                                 mode='lufs' if body.get('mode') == 'lufs' else 'peak', peak_db=float(body.get('peak', -1)),
                                 lufs=float(body.get('lufs', -18)), pattern=body.get('pattern') or '{category}_{name}_{nn}',
                                 category=(body.get('category') or '').strip() or None)
        return {'out': str(out.resolve()), 'files': [r['output_file'] for r in rows], 'manifest': str(out.resolve() / 'MANIFEST.csv')}

    def verdicts(self):
        try:
            return json.loads(self.verdicts_path.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def set_verdict(self, rel, verdict, note):
        with self.lock:
            data = self.verdicts()
            if verdict or note:
                data[rel] = {'verdict': verdict, 'note': note, 'at': datetime.now().isoformat(timespec='seconds')}
            else:
                data.pop(rel, None)
            tmp = self.verdicts_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding='utf-8')
            tmp.replace(self.verdicts_path)
        return data

    def file(self, rel):
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise FileNotFoundError(rel)
        return path


def handler(lib):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, data, status=200):
            body = json.dumps(data, separators=(',', ':')).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == '/':
                    body = (HERE / 'sound_browser.html').read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == '/api/data':
                    self.send_json(lib.payload())
                elif url.path == '/api/search':
                    self.send_json(lib.search(q['q']))
                elif url.path == '/api/approved':
                    self.send_json(lib.more_like(avoid=q.get('avoid', '')))
                elif url.path == '/api/like':
                    self.send_json(lib.more_like(path=q['p'], avoid=q.get('avoid', '')))
                elif url.path == '/api/varied':
                    self.send_json(lib.varied(q['p'], int(q.get('n', 4))))
                elif url.path == '/api/similar':
                    self.send_json(lib.similar(int(q['row'])))
                elif url.path == '/api/waveform':
                    self.send_json(lib.waveform(q['p']))
                elif url.path == '/api/spectrogram':
                    body = lib.spectrogram(q['p']).read_bytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/png')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == '/audio':
                    self.send_audio(lib.file(q['p']))
                else:
                    self.send_json({'error': 'not found'}, 404)
            except (ConnectionError, BrokenPipeError):
                pass  # the page dropped a stream when the user skipped ahead
            except ValueError as exc:
                self.send_json({'error': str(exc)}, 400)
            except FileNotFoundError as exc:
                self.send_json({'error': f'missing: {exc}'}, 404)
            except Exception as exc:
                self.send_json({'error': str(exc)}, 500)

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
                if self.path == '/api/verdict':
                    lib.file(body['path'])  # only real library files get a verdict
                    self.send_json(lib.set_verdict(body['path'], body.get('verdict', ''), body.get('note', '')))
                elif self.path == '/api/export':
                    self.send_json(lib.export(body))
                elif self.path == '/api/compare':
                    self.send_json(lib.compare(body.get('paths')))
                elif self.path == '/api/choose-folder':
                    self.send_json({'folder': choose_folder()})
                elif self.path == '/api/reveal':
                    path = lib.file(body['path'])
                    if os.name == 'nt':
                        command = ['explorer', '/select,', str(path)]
                    elif sys.platform == 'darwin':
                        command = ['open', '-R', str(path)]
                    else:
                        command = ['xdg-open', str(path.parent)]
                    subprocess.Popen(command)
                    self.send_json({'ok': True})
                else:
                    self.send_json({'error': 'not found'}, 404)
            except ValueError as exc:
                self.send_json({'error': str(exc)}, 400)
            except FileNotFoundError as exc:
                self.send_json({'error': f'missing: {exc}'}, 404)
            except Exception as exc:
                self.send_json({'error': str(exc)}, 500)

        def send_audio(self, path):
            """Byte ranges, so the player can seek inside long files."""
            size = path.stat().st_size
            start, end = 0, size - 1
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', self.headers.get('Range', ''))
            if match and (match[1] or match[2]):
                start = int(match[1]) if match[1] else max(0, size - int(match[2]))
                end = min(int(match[2]), size - 1) if match[1] and match[2] else size - 1
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            else:
                self.send_response(200)
            self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            self.end_headers()
            with path.open('rb') as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

    return Handler


def choose_folder():
    """The system's folder picker, when Python has Tk; '' when cancelled. The page also accepts a typed path."""
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError as exc:
        raise ValueError('No folder picker here (Python without Tk): type the folder path instead') from exc
    window = tkinter.Tk()
    window.withdraw()
    window.attributes('-topmost', True)
    try:
        return filedialog.askdirectory(title='Export sounds to') or ''
    finally:
        window.destroy()


class Server(ThreadingHTTPServer):
    # HTTPServer sets SO_REUSEADDR, which on Windows lets a second server bind a port already in use:
    # two servers would share it silently and the free-port fallback below would never trigger.
    allow_reuse_address = False


def main():
    parser = argparse.ArgumentParser(description='Browse, audition, compare and export a tagged sound library locally.',
                                     epilog='Example: python scripts/sound_browser.py --root D:/MySounds --port 8790',
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='library folder holding AUDIO_TAGS.csv')
    parser.add_argument('--port', type=int, default=8790, help='first localhost port to try (default: 8790; tries 20 ports)')
    parser.add_argument('--no-open', action='store_true', help='serve without opening a browser window')
    a = parser.parse_args()
    lib = Library(a.root)
    for port in range(a.port, a.port + 20):  # another local tool may hold the port
        try:
            server = Server(('127.0.0.1', port), handler(lib))
            break
        except OSError:
            continue
    else:
        sys.exit(f'No free port from {a.port} to {a.port + 19}')
    port = server.server_address[1]
    url = f'http://127.0.0.1:{port}/'
    print(f'Sound browser on {url} for {lib.root} - Ctrl+C to stop', flush=True)
    if not a.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

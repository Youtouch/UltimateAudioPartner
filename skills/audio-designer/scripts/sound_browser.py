"""Browse a sound library by what it contains: tags, description search, "more like this", audition.

    python sound_browser.py --root D:/MySounds       # opens the browser
    python sound_browser.py --root D:/MySounds --port 8790 --no-open

Runs only while this window is open (Ctrl+C stops it) and listens on 127.0.0.1 only.
Reads AUDIO_LEDGER_files.csv, AUDIO_TAGS.csv and AUDIO_TAGS_clap.f32 from the root; writes
your likes, rejects and notes to AUDIO_VERDICTS.json there, where agents read them too.
"""

import argparse
import csv
import json
import mimetypes
import re
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

HERE = Path(__file__).resolve().parent
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

    def payload(self):
        """Ledger rows joined with the latest tag row per path. Untagged files are listed too."""
        ledger = {}
        path = self.root / 'AUDIO_LEDGER_files.csv'
        if path.exists():
            with path.open(encoding='utf-8-sig', newline='') as f:
                for r in csv.DictReader(f):
                    ledger[r['relative_path']] = r
        tagged = {}
        if self.tags.exists():
            with self.tags.open(encoding='utf-8', newline='') as f:
                for r in csv.DictReader(f):
                    tagged[r['relative_path']] = r  # last row wins: a changed file appends a new one
        names, index, rows = [], {}, []

        def ids(items):
            out = []
            for name, score in items:
                if name not in index:
                    index[name] = len(names)
                    names.append(name)
                out += [index[name], score]
            return out

        for rel in sorted(set(ledger) | set(tagged)):
            led, tag = ledger.get(rel, {}), tagged.get(rel)
            timeline = []
            if tag:
                for part in tag['timeline'].split(' | '):
                    span, _, labels = part.partition(': ')
                    if span:
                        timeline.append([float(span.split('-')[0]), ids(pairs(labels))])
            rows.append([rel, led.get('pack') or rel.split('/')[0], led.get('bucket') or '',
                         float(led.get('duration_seconds') or 0) or None,
                         float(tag['speech']) if tag else None, float(tag['music']) if tag else None,
                         ids(pairs(tag['top_labels'])) if tag else [], timeline,
                         int(tag['clap_row']) if tag and tag.get('clap_row') else -1])
        return {'labels': names, 'rows': rows, 'verdicts': self.verdicts(), 'root': str(self.root),
                'tagged': len(tagged), 'total': len(rows)}

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

    def search(self, query):
        """A description: audio, name and label sources fused (audio_search.py). Rebuilt when the tags change."""
        sys.path.insert(0, str(HERE))
        from audio_search import Index  # first call loads CLAP and embeds new names
        stamp = self.tags.stat().st_mtime_ns
        with self.lock:
            # A running tagging pass rewrites the CSV constantly: rebuild at most every 10 minutes.
            stale = getattr(self, '_index_stamp', None) != stamp and time.time() - getattr(self, '_index_built', 0) > 600
            if getattr(self, '_index', None) is None or stale:
                self._index, self._index_stamp, self._index_built = Index(self.root), stamp, time.time()
            index = self._index
        results, wanted = index.find(query, top=1000)
        scores = [None] * len(self.matrix())
        evidence = {}
        for r in results:
            if r['row'] < len(scores):
                scores[r['row']] = r['score']
                evidence[r['row']] = r['evidence']
        return {'scores': scores, 'evidence': evidence, 'labels': list(wanted)}

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
                elif url.path == '/api/similar':
                    self.send_json(lib.similar(int(q['row'])))
                elif url.path == '/audio':
                    self.send_audio(lib.file(q['p']))
                else:
                    self.send_json({'error': 'not found'}, 404)
            except (ConnectionError, BrokenPipeError):
                pass  # the page dropped a stream when the user skipped ahead
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
                elif self.path == '/api/reveal':
                    subprocess.Popen(['explorer', '/select,', str(lib.file(body['path']))])
                    self.send_json({'ok': True})
                else:
                    self.send_json({'error': 'not found'}, 404)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='library folder holding AUDIO_TAGS.csv')
    parser.add_argument('--port', type=int, default=8790)
    parser.add_argument('--no-open', action='store_true')
    a = parser.parse_args()
    lib = Library(a.root)
    for port in range(a.port, a.port + 20):  # another local tool may hold the port
        try:
            server = ThreadingHTTPServer(('127.0.0.1', port), handler(lib))
            break
        except OSError:
            continue
    else:
        sys.exit(f'No free port from {a.port} to {a.port + 19}')
    url = f'http://127.0.0.1:{port}/'
    print(f'Sound browser on {url} for {lib.root} - Ctrl+C to stop')
    if not a.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()

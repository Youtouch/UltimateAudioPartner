"""Verify the public demo guide, dated measurements and compact screenshot assets."""
import argparse
import json
from pathlib import Path
import struct


def contract(root):
    readme = (root / 'README.md').read_text(encoding='utf-8')
    assert 'python tools/demo.py' in readme[:2500], 'Demo missing from README quick start'
    for name in ['library', 'sound-detail', 'explain', 'compare', 'export', 'category-recall']:
        relative = 'docs/img/' + name + '.png'
        assert relative in readme, 'Screenshot missing from feature description: ' + relative
        body = (root / relative).read_bytes()
        assert body[:8] == b'\x89PNG\r\n\x1a\n', 'Not a PNG: ' + relative
        assert len(body) < 300_000, 'Screenshot exceeds 300 KB: ' + relative
        width, height = struct.unpack('>II', body[16:24])
        assert width >= 1000 and height >= 700, 'Screenshot too small: ' + relative
    evidence = json.loads((root / 'docs/img/capture-evidence.json').read_text(encoding='utf-8'))
    assert evidence['synthetic'] and 'Microsoft Edge headless' in evidence['engine']
    assert {r['file'] for r in evidence['screenshots']} == {'library.png', 'sound-detail.png', 'explain.png', 'compare.png', 'export.png', 'category-recall.png'}
    assert 'Source agreement and filenames do not' in readme and 'listening precision or recall' in readme
    assert '513 filename-sampled' not in readme and '| DEV | 11,199' not in readme, 'Private benchmark results published'
    assert 'AUDIO_VISUALS/' in readme, 'Visual cache missing from file reference'
    how = (root / 'docs/how-it-works.md').read_text(encoding='utf-8')
    assert how.count('```mermaid') == 1, 'How it works needs one Mermaid diagram'
    assert all(word in how.lower() for word in ['ledger', 'descriptor', 'fingerprint', 'agreement', 'export', 'render', 'check'])
    assert not (root / 'skills/audio-designer/bench').exists(), 'Research was published'
    assert '59' in (root / 'CHANGELOG.md').read_text(encoding='utf-8') or 'CED' in (root / 'CHANGELOG.md').read_text(encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__, epilog='Example: python tools/polish_tests.py')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1], help='repository to inspect')
    args = parser.parse_args()
    contract(args.root)
    print('PASS public demo, six compact screenshots, measured dates and explanation diagram')


if __name__ == '__main__':
    main()

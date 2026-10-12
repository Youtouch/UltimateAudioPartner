"""Model-free publication regressions; fixtures are temporary and subprocesses are mocked.

Run with python tools/publication_tests.py. No library, models or Git are used.
"""
import contextlib
import io
from pathlib import Path
import shutil
import subprocess
import struct
import sys
import tempfile
import zlib
from unittest.mock import patch

import check_all
import sync_from_live as sync


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def source_fixture(root):
    # The root name intentionally differs from the installed skill's usual name.
    source = root / 'renamed-source'
    files = {
        'SKILL.md': 'Public method.\n',
        'scripts/audio_map.py': '# public script\n',
        'references/mix-plan.md': 'Public mix plan.\n',
        'bench/README.md': 'Public benchmark explanation.\n',
        'night_report.md': 'A private report must never be copied.\n',
        'personal/journal.txt': 'Private notes.\n',
        'scripts/private_notes.txt': 'Private script notes.\n',
        'scripts/unlisted_helper.py': '# unlisted private helper\n',
        'references/personal.md': 'Private reference.\n',
        'bench/unlisted.py': '# private experiment\n',
        'tagging/unlisted.py': '# private experiment\n',
        'mix/REPORT.md': 'Private mix report.\n',
        'render/report.txt': 'Private render report.\n',
        'scripts/__pycache__/cached.pyc': 'Not public source.\n',
    }
    for relative, text in files.items():
        write(source / relative, text)
    return source


def public_copy_contract(copy=sync.copy_skill):
    with tempfile.TemporaryDirectory(prefix='uap-publication-test-') as temporary:
        root = Path(temporary)
        source = source_fixture(root)
        destination = root / 'published'
        copy(source, destination)
        actual = {path.relative_to(destination).as_posix() for path in destination.rglob('*') if path.is_file()}
        expected = {'SKILL.md', 'scripts/audio_map.py', 'references/mix-plan.md', 'bench/README.md'}
        assert actual == expected, f'Publication allowlist leaked or omitted files: {sorted(actual ^ expected)}'


def privacy_contract(check=sync.privacy_check):
    with tempfile.TemporaryDirectory(prefix='uap-privacy-test-') as temporary:
        root = Path(temporary)
        stage = root / 'stage'
        words = root / 'words.txt'
        token = 'orchid secret phrase'
        write(words, '# comment\n\n' + token + '\n')
        full_line = 'prefix ORCHID SECRET PHRASE suffix'
        write(stage / 'guide.md', 'A public first line.\n' + full_line + '\n')
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            failures = check(stage, words)
        assert failures == [('guide.md', 2, token)], failures
        assert full_line not in output.getvalue(), 'Privacy diagnostics exposed the matching source line'
        write(stage / 'guide.md', 'Public text only.\n')
        with contextlib.redirect_stdout(io.StringIO()):
            assert check(stage, words) == [], 'Clean publication was rejected'


def screenshot_copy_contract():
    with tempfile.TemporaryDirectory(prefix='uap-images-test-') as temporary:
        root = Path(temporary)
        source, target = root / 'source', root / 'copy'
        write(source / 'CHANGELOG.md', 'Public changes.\n')
        for name in ('library.png', 'sound-detail.png', 'explain.png', 'compare.png', 'export.png', 'private.png'):
            path = source / 'docs/img' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'\x89PNG\r\n\x1a\n')
        sync.copy_repo_files(source, target)
        assert (target / 'CHANGELOG.md').is_file(), 'Changelog omitted from publication'
        copied = {p.name for p in (target / 'docs/img').glob('*')}
        assert copied == {'library.png', 'sound-detail.png', 'explain.png', 'compare.png', 'export.png'}, copied


def png_privacy_contract():
    def chunk(kind, body):
        return struct.pack('>I', len(body)) + kind + body + struct.pack('>I', zlib.crc32(kind + body))
    def png(metadata=b''):
        return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
                + metadata + chunk(b'IDAT', zlib.compress(b'\0\x17\x37\x5b')) + chunk(b'IEND', b''))
    with tempfile.TemporaryDirectory(prefix='uap-png-privacy-') as temporary:
        root = Path(temporary)
        stage, words = root / 'stage', root / 'words.txt'
        stage.mkdir()
        (stage / 'screen.png').write_bytes(png())
        # A substring of the compressed image is not visible text or metadata.
        raw = (stage / 'screen.png').read_bytes()
        binary_fragment = raw[raw.index(b'IDAT') + 4:][:4].decode('latin-1')
        write(words, binary_fragment + '\n')
        with contextlib.redirect_stdout(io.StringIO()):
            assert sync.privacy_check(stage, words) == [], 'Compressed pixels scanned as prose'
        write(words, 'orchid secret phrase\n')
        for kind in (b'tEXt', b'zTXt', b'iTXt'):
            secret = b'ORCHID SECRET PHRASE'
            body = b'Comment\0' + (b'\0' + zlib.compress(secret) if kind == b'zTXt' else
                                   b'\1\0\0\0' + zlib.compress(secret) if kind == b'iTXt' else secret)
            (stage / 'screen.png').write_bytes(png(chunk(kind, body)))
            with contextlib.redirect_stdout(io.StringIO()):
                assert sync.privacy_check(stage, words), 'PNG metadata bypassed privacy gate'
        (stage / 'screen.png').write_text('ORCHID SECRET PHRASE', encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            assert sync.privacy_check(stage, words), 'Renaming text as PNG bypassed privacy gate'


def tree_snapshot(root):
    if not root.exists():
        return None
    result = {}
    for path in [root, *sorted(root.rglob('*'))]:
        key = path.relative_to(root).as_posix()
        stat = path.stat()
        result[key] = (path.is_dir(), stat.st_mtime_ns, stat.st_mode,
                       None if path.is_dir() else path.read_bytes())
    return result


def gate_fixture(root, blocked=False, destination_exists=True):
    repo = root / 'source-repo'
    write(repo / 'README.md', 'Public project.\n')
    write(repo / 'tools/check_all.py', '# runner is mocked\n')
    write(repo / 'tools/smoke_test.py', '# smoke runner is mocked\n')
    write(repo / 'tools/private_words.txt', 'orchid secret phrase\n')
    source = source_fixture(root)
    if blocked:
        write(source / 'SKILL.md', 'This contains ORCHID SECRET PHRASE.\n')
    destination = root / 'destination'
    if destination_exists:
        write(destination / 'skills/audio-designer/SKILL.md', 'Original installed skill.\n')
        write(destination / 'keep.txt', 'Original destination marker.\n')
    return repo, source, destination


def gate_contract(gate, destination_exists=True):
    with tempfile.TemporaryDirectory(prefix='uap-gate-test-') as temporary:
        root = Path(temporary)
        repo, source, destination = gate_fixture(root, blocked=gate == 'privacy',
                                                destination_exists=destination_exists)
        before = tree_snapshot(destination)
        command = ['sync_from_live.py', '--from', str(source), '--dest', str(destination)]
        status = 0 if gate == 'privacy' else 7
        with patch.object(sync, '__file__', str(repo / 'tools/sync_from_live.py')), \
                patch.object(sys, 'argv', command), contextlib.redirect_stdout(io.StringIO()), \
                patch.object(sync.subprocess, 'run', return_value=subprocess.CompletedProcess([], status)) as run:
            result = sync.main()
        assert result != 0, f'{gate} failure did not block publication'
        assert tree_snapshot(destination) == before, f'{gate} failure changed the destination'
        if gate == 'privacy':
            run.assert_not_called()
        else:
            run.assert_called_once()
            args = run.call_args.args[0]
            assert Path(args[2]).name == 'check_all.py' and '--published' in args, args


def successful_gate_contract():
    with tempfile.TemporaryDirectory(prefix='uap-success-test-') as temporary:
        root = Path(temporary)
        repo, source, destination = gate_fixture(root)
        before = tree_snapshot(destination)
        def passed(args, **kwargs):
            assert tree_snapshot(destination) == before, 'Destination changed before the test gate finished'
            return subprocess.CompletedProcess(args, 0)
        with patch.object(sync, '__file__', str(repo / 'tools/sync_from_live.py')), \
                patch.object(sys, 'argv', ['sync_from_live.py', '--from', str(source), '--dest', str(destination)]), \
                patch.object(sync.subprocess, 'run', side_effect=passed) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            assert sync.main() == 0
        run.assert_called_once()
        published = destination / 'skills/audio-designer'
        assert (published / 'SKILL.md').read_bytes() == (source / 'SKILL.md').read_bytes()
        assert not (published / 'night_report.md').exists()
        assert (destination / 'keep.txt').read_text(encoding='utf-8') == 'Original destination marker.\n'


def runner_contract(failure=None):
    with tempfile.TemporaryDirectory(prefix='uap-runner-test-') as temporary:
        root = Path(temporary)
        repo, source, _ = gate_fixture(root)
        calls = []
        def simulate(args, **kwargs):
            name = Path(args[2]).name
            calls.append(name)
            is_smoke = name == 'smoke_test.py'
            if is_smoke:
                assert '--quick' in args and '--published' in args, args
            failed = ((failure == 'selftest' and len(calls) == 1)
                      or (failure == 'smoke' and is_smoke))
            return subprocess.CompletedProcess(args, int(failed), stdout='', stderr='')
        with patch.object(check_all, '__file__', str(repo / 'tools/check_all.py')), \
                patch.object(sys, 'argv', ['check_all.py', '--from', str(source)]), \
                patch.object(check_all.subprocess, 'run', side_effect=simulate), \
                contextlib.redirect_stdout(io.StringIO()):
            result = check_all.main()
        assert bool(result) == bool(failure), f'check_all lost {failure or "success"} status: {result}'
        assert calls.count('smoke_test.py') == 1 and len(calls) >= len(check_all.CHECKS) + 1, calls


def expect_rejected(name, action):
    try:
        action()
    except AssertionError:
        print(f'PASS publication mutation rejected: {name}')
    else:
        raise AssertionError(f'Regression did not reject mutation: {name}')


def mutation_checks():
    def basename_only(source, destination):
        # Reproduce the old copy implementation, whose root filtering depended on its basename.
        shutil.copytree(source, destination, ignore=sync.ignore)
    expect_rejected('renamed source bypasses root allowlist', lambda: public_copy_contract(basename_only))
    expect_rejected('privacy scanner disabled', lambda: privacy_contract(lambda root, words: []))
    with patch.object(sync, 'privacy_check', return_value=[]):
        expect_rejected('publication proceeds without privacy gate', lambda: gate_contract('privacy'))


def main():
    if not __debug__:
        raise RuntimeError('Self-tests require asserts: do not use python -O')
    tests = [
        ('renamed-source publication allowlist', public_copy_contract),
        ('case-insensitive privacy blockers and safe diagnostics', privacy_contract),
        ('only named public screenshots and changelog are copied', screenshot_copy_contract),
        ('PNG pixels stay binary and text metadata is privacy checked', png_privacy_contract),
        ('existing destination survives privacy failure', lambda: gate_contract('privacy')),
        ('absent destination survives privacy failure', lambda: gate_contract('privacy', False)),
        ('existing destination survives test-runner failure', lambda: gate_contract('tests')),
        ('absent destination survives test-runner failure', lambda: gate_contract('tests', False)),
        ('publication succeeds only after passing gates', successful_gate_contract),
        ('check_all successful status', runner_contract),
        ('check_all self-test failure propagates', lambda: runner_contract('selftest')),
        ('check_all smoke failure propagates', lambda: runner_contract('smoke')),
        ('old publication defects rejected', mutation_checks),
    ]
    failures = []
    for name, action in tests:
        try:
            action()
        except Exception as exc:
            failures.append(name)
            print(f'FAIL publication: {name}: {type(exc).__name__}: {exc}', file=sys.stderr)
        else:
            print(f'PASS publication: {name}')
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())

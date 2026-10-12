"""Check the first-run synthetic demo without loading any recognition models.

Example: python tools/demo_tests.py
"""
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def contract(root, source):
    assert (source / 'tools/demo.py').is_file(), 'missing first-run demo command'
    spec = importlib.util.spec_from_file_location('uap_demo', source / 'tools/demo.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    skill = demo.skill_source(source)
    sys.path.insert(0, str(skill / 'scripts'))
    import audio_tag
    import audio_search
    import demo_metadata
    with patch.object(audio_tag, 'embed_text', side_effect=AssertionError('demo loaded CLAP')), \
            patch.object(audio_tag, 'tagger', side_effect=AssertionError('demo loaded tagger')):
        library = demo.build_demo(root / 'library', source=skill, workers=1)
        marker = demo_metadata.read_demo(library)
        assert marker and marker['schema'] == demo_metadata.SCHEMA, 'demo provenance missing'
        assert 'generator' in marker['labels'] and 'not recognition' in marker['labels'], marker
        tags = read_csv(library / 'AUDIO_TAGS.csv')
        assert len(tags) >= 13 and all(t['tagger'] == 'synthetic-generator-v1' for t in tags), tags
        names = {t['relative_path'] for t in tags}
        required = {'Game SFX/Coin/coin_chirp_01.wav', 'Game SFX/Laser/laser_sweep_01.wav',
                    'Game SFX/Footsteps/footstep_thud_01.wav', 'Game SFX/UI/ui_click_01.wav'}
        assert required <= names, 'game effects missing'
        assert not any('corrupt' in p for p in names), 'demo contains smoke-test corruption'
        descriptors = read_csv(library / 'AUDIO_DESCRIPTORS.csv')
        assert {r['relative_path'] for r in descriptors} == names, 'demo did not measure every sound'
        ledger = read_csv(library / 'AUDIO_LEDGER_files.csv')
        assert {r['relative_path'] for r in ledger} == names, 'demo ledger missing'
        assert max(float(r['duration_seconds']) for r in ledger) <= 10, 'demo retained 90-second smoke fixtures'
        vocab = read_csv(library / 'AUDIO_VOCAB_TAGS.csv')
        assert vocab and all(r['name'] == '1' and not r['label'] and not r['audio'] for r in vocab), \
            'generated labels incorrectly claimed independent agreement'
        index = audio_search.Index(library)
        found, _ = index.find('short bright click', top=20)
        assert found and any('click' in r['path'].lower() for r in found[:4]), found
        assert all(set(r['evidence']) == {'name'} for r in found), 'demo search claimed model evidence'
        similar = index.audio @ index.audio[0]
        assert len(similar) == len(tags) and abs(float(similar[0]) - 1) < 1e-5
        # A normal library still takes the existing model path. A marker must never change other roots.
        (library / 'UAP_DEMO.json').unlink()
        assert demo_metadata.read_demo(library) is None
        with patch.object(audio_tag, 'embed_text', side_effect=RuntimeError('ordinary model path')):
            try:
                audio_search.Index(library)
            except RuntimeError as exc:
                assert str(exc) == 'ordinary model path'
            else:
                raise AssertionError('demo fallback escaped its marked synthetic library')
    print('PASS demo: smoke generator + four game effects, complete measurements, offline name search, single-source provenance')


def launcher_contract(source):
    for name in ('Tag Library.bat', 'Sound Browser.bat', 'tag-library.sh', 'sound-browser.sh'):
        text = (source / name).read_text(encoding='utf-8')
        assert 'demo' in text and 'tools' in text and 'demo.py' in text, f'{name} lacks demo routing'
    with tempfile.TemporaryDirectory(prefix='uap-demo-launchers-') as temporary:
        root = Path(temporary)
        checkout = root / 'checkout with spaces'
        (checkout / 'tools').mkdir(parents=True)
        log = root / 'calls.txt'
        for name in ('Tag Library.bat', 'Sound Browser.bat', 'tag-library.sh', 'sound-browser.sh'):
            shutil.copy2(source / name, checkout / name)
        stub = ("import os, pathlib\n"
                "with pathlib.Path(os.environ['UAP_DEMO_LOG']).open('a') as out: out.write('demo\\n')\n"
                "raise SystemExit(int(os.environ.get('UAP_DEMO_STATUS', '0')))\n")
        (checkout / 'tools/demo.py').write_text(stub, encoding='utf-8')
        env = dict(os.environ, UAP_DEMO_LOG=str(log),
                   PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''))
        if os.name == 'nt':
            for name in ('Tag Library.bat', 'Sound Browser.bat'):
                for status in (0, 9):
                    log.write_text('', encoding='utf-8')
                    command = f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /s /c ""{checkout / name}" demo"'
                    run = subprocess.run(command, cwd=root, env={**env, 'UAP_DEMO_STATUS': str(status)},
                                         capture_output=True, encoding='utf-8', errors='replace')
                    assert run.returncode == status and log.read_text().splitlines() == ['demo'], \
                        f'{name} ignored demo status {status}: {run.stdout} {run.stderr}'
        bash = shutil.which('bash')
        if not bash and os.name == 'nt':
            candidate = Path(os.environ.get('ProgramFiles', '')) / 'Git/bin/bash.exe'
            bash = str(candidate) if candidate.is_file() else None
        if bash:
            binary = root / 'bin'
            binary.mkdir()
            stub_path = binary / 'python3'
            stub_path.write_text('#!/usr/bin/env bash\nprintf "demo\\n" >> "$UAP_DEMO_LOG"\n'
                                 'exit "${UAP_DEMO_STATUS:-0}"\n', encoding='utf-8')
            stub_path.chmod(0o755)
            for name in ('tag-library.sh', 'sound-browser.sh'):
                for status in (0, 9):
                    log.write_text('', encoding='utf-8')
                    run = subprocess.run([bash, (checkout / name).as_posix(), 'demo'], cwd=root,
                                         env={**env, 'PATH': str(binary) + os.pathsep + env['PATH'],
                                              'UAP_DEMO_LOG': log.as_posix(), 'UAP_DEMO_STATUS': str(status)},
                                         capture_output=True, encoding='utf-8', errors='replace')
                    assert run.returncode == status and log.read_text().splitlines() == ['demo'], \
                        f'{name} ignored demo status {status}: {run.stdout} {run.stderr}'
        assert not (checkout / 'library.txt').exists(), 'demo replaced the remembered user library'
    print('PASS demo launchers: native tag/browser routing, spaces, failure status, no library prompt or saved-path change')


def main():
    with tempfile.TemporaryDirectory(prefix='uap-demo-tests-') as temporary:
        contract(Path(temporary), REPO)
    launcher_contract(REPO)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.SubprocessError) as error:
        print('FAIL demo: ' + str(error), file=sys.stderr)
        raise SystemExit(1)

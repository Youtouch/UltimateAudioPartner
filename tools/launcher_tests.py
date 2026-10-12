"""Exercise launcher paths and failure propagation with synthetic command stubs."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def copy_launchers(destination):
    (destination / 'tools').mkdir(parents=True)
    for relative in ('tag-library.sh', 'sound-browser.sh', 'Tag Library.bat', 'Sound Browser.bat',
                     'tools/library.sh', 'tools/library.bat'):
        shutil.copy2(REPO / relative, destination / relative)


def shell_checks(bash, root):
    checkout = root / 'shell checkout'
    copy_launchers(checkout)
    first, other = root / 'first cwd', root / 'other cwd'
    first.mkdir()
    other.mkdir()
    library = first / 'library with spaces é 日本 & rain'
    library.mkdir()
    binary = root / 'shell bin'
    binary.mkdir()
    stub = binary / 'python3'
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$UAP_LAUNCH_LOG"\n'
                    'if [ -z "${UAP_FAIL_STAGE:-}" ] || [ "${1##*/}" = "$UAP_FAIL_STAGE" ]; then\n'
                    '  exit "${UAP_LAUNCH_STATUS:-0}"\nfi\nexit 0\n', encoding='utf-8')
    stub.chmod(0o755)
    log = root / 'shell calls.txt'
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ.get('PATH', ''),
               UAP_LAUNCH_LOG=log.as_posix(), UAP_LAUNCH_STATUS='0')
    env.pop('AUDIO_LIBRARY', None)

    def run(launcher, cwd, argument=None, status=0, stage=''):
        log.write_text('', encoding='utf-8')
        args = [str(bash), (checkout / launcher).as_posix()]
        if argument is not None:
            args.append(argument)
        result = subprocess.run(args, cwd=cwd, env={**env, 'UAP_LAUNCH_STATUS': str(status), 'UAP_FAIL_STAGE': stage},
                                input='\n', capture_output=True, encoding='utf-8', errors='replace')
        return result, log.read_text(encoding='utf-8').splitlines()

    result, calls = run('tag-library.sh', first, library.name)
    require(result.returncode == 0 and len(calls) == 3, 'shell tag stages failed: ' + result.stderr)
    result, calls = run('tag-library.sh', other)
    require(result.returncode == 0 and len(calls) == 3, 'saved relative library broke from another cwd')
    saved = checkout / 'library.txt'
    saved.write_bytes(saved.read_bytes().replace(b'\n', b'\r\n'))
    result, calls = run('sound-browser.sh', other)
    require(result.returncode == 0 and len(calls) == 1, 'shell could not load CRLF library.txt')
    for launcher in ('tag-library.sh', 'sound-browser.sh'):
        result, calls = run(launcher, first, library.name, status=7)
        require(result.returncode == 7 and len(calls) == 1,
                f'{launcher} ignored command failure: exit {result.returncode}, {len(calls)} stages')
    for count, stage in enumerate(('audio_tag.py', 'audio_search.py', 'vocab_tags.py'), 1):
        result, calls = run('tag-library.sh', first, library.name, status=7, stage=stage)
        require(result.returncode == 7 and len(calls) == count, f'shell did not stop after failed {stage}')
    # Prove the changed-path assertion rejects the original relative-path behavior.
    helper = checkout / 'tools/library.sh'
    source = helper.read_text(encoding='utf-8')
    helper.write_text(source.replace('LIBRARY="$(cd -- "$LIBRARY" && pwd -P)"\n', ''), encoding='utf-8')
    run('tag-library.sh', first, library.name)
    result, _ = run('tag-library.sh', other)
    require(result.returncode != 0, 'shell path regression accepted the original broken behavior')
    mutations = 1
    if os.name != 'nt':
        # Win32 path APIs tolerate a trailing CR; native Unix directory lookup does not.
        helper.write_text(source.replace('LIBRARY="${LIBRARY%$\'\\r\'}"\n', ''), encoding='utf-8')
        run('tag-library.sh', first, library.name)
        saved.write_bytes(saved.read_bytes().replace(b'\n', b'\r\n'))
        result, _ = run('sound-browser.sh', other)
        require(result.returncode != 0, 'shell CRLF regression accepted the original broken behavior')
        mutations += 1
    print(f'PASS shell launchers: Unicode/spaces, saved absolute paths, CRLF, failure propagation; {mutations} mutation(s) rejected')


def windows_checks(root):
    checkout = root / 'batch checkout'
    copy_launchers(checkout)
    first, other = root / 'batch first cwd', root / 'batch other cwd'
    first.mkdir()
    other.mkdir()
    library = first / 'library with spaces & rain'
    library.mkdir()
    log = root / 'batch calls.jsonl'
    stub = ("import json, os, sys\n"
            "with open(os.environ['UAP_LAUNCH_LOG'], 'a', encoding='utf-8') as out:\n"
            "    out.write(json.dumps(sys.argv, ensure_ascii=False) + '\\n')\n"
            "stage = os.environ.get('UAP_FAIL_STAGE', '')\n"
            "raise SystemExit(int(os.environ.get('UAP_LAUNCH_STATUS', '0')) if not stage or os.path.basename(sys.argv[0]) == stage else 0)\n")
    for relative in ('scripts/audio_tag.py', 'scripts/audio_search.py', 'tagging/vocab_tags.py',
                     'scripts/sound_browser.py'):
        target = checkout / 'skills/audio-designer' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(stub, encoding='utf-8')
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', ''),
               UAP_LAUNCH_LOG=str(log))
    env.pop('AUDIO_LIBRARY', None)

    def run(launcher, cwd, argument=None, status=0, stage=''):
        log.write_text('', encoding='utf-8')
        argument = f' "{argument}"' if argument is not None else ''
        command = f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /s /c ""{checkout / launcher}"{argument}"'
        result = subprocess.run(command, cwd=cwd, env={**env, 'UAP_LAUNCH_STATUS': str(status), 'UAP_FAIL_STAGE': stage},
                                input='\n', capture_output=True, encoding='utf-8', errors='replace')
        return result, [json.loads(line) for line in log.read_text(encoding='utf-8').splitlines()]

    result, calls = run('Tag Library.bat', first, library.name)
    require(result.returncode == 0 and len(calls) == 3, 'batch tag stages failed: ' + result.stderr)
    result, calls = run('Tag Library.bat', other)
    require(result.returncode == 0 and len(calls) == 3, 'saved batch path broke from another cwd')
    for launcher in ('Tag Library.bat', 'Sound Browser.bat'):
        result, calls = run(launcher, first, library.name, status=7)
        require(result.returncode == 7 and len(calls) == 1,
                f'{launcher} ignored failure: exit {result.returncode}, {len(calls)} stages; {result.stderr}')
    for count, stage in enumerate(('audio_tag.py', 'audio_search.py', 'vocab_tags.py'), 1):
        result, calls = run('Tag Library.bat', first, library.name, status=7, stage=stage)
        require(result.returncode == 7 and len(calls) == count, f'batch did not stop after failed {stage}')
    launcher = checkout / 'Tag Library.bat'
    source = launcher.read_text(encoding='utf-8')
    launcher.write_text(source.replace(' || goto :failed', ''), encoding='utf-8')
    result, calls = run('Tag Library.bat', first, library.name, status=7)
    require(result.returncode != 7 or len(calls) != 1, 'batch regression accepted ignored command failures')
    print('PASS batch launchers: spaces/metacharacters, saved absolute paths, exit codes; failure mutation rejected')


def main():
    bash = shutil.which('bash')
    if not bash and os.name == 'nt':
        candidate = Path(os.environ.get('ProgramFiles', '')) / 'Git/bin/bash.exe'
        bash = str(candidate) if candidate.is_file() else None
    with tempfile.TemporaryDirectory(prefix='uap-launchers-') as temporary:
        root = Path(temporary)
        if bash:
            shell_checks(bash, root)
        else:
            print('SKIP shell launcher runtime: Bash is unavailable')
        if os.name == 'nt':
            windows_checks(root)
        else:
            print('SKIP batch launcher runtime: requires Windows cmd.exe')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.SubprocessError) as error:
        print('FAIL launchers: ' + str(error), file=sys.stderr)
        raise SystemExit(1)

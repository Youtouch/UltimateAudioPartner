"""Check every public CLI's description, arguments and executable help examples."""

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

from sync_from_live import PUBLISH_BENCH, PUBLISH_MIX, PUBLISH_RENDER, PUBLISH_SCRIPTS, PUBLISH_TAGGING


def public_clis(repo, skill):
    """Discover published argparse entry points so new commands enter the gate."""
    candidates = [*sorted((repo / 'tools').glob('*.py'))]
    for folder, names in [('scripts', PUBLISH_SCRIPTS), ('bench', PUBLISH_BENCH),
                          ('tagging', PUBLISH_TAGGING), ('mix', PUBLISH_MIX), ('render', PUBLISH_RENDER)]:
        candidates.extend(skill / folder / name for name in sorted(names) if name.endswith('.py'))
    found = []
    for path in candidates:
        if not path.is_file():
            continue  # An explicit older --from snapshot may predate a registered addition.
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == 'ArgumentParser' for node in ast.walk(tree)):
            found.append(path)
    return found


class ParserCaptured(BaseException):
    pass


def capture_parser(path):
    captured = []
    def capture(parser, *args, **kwargs):
        captured.append(parser)
        raise ParserCaptured()
    # Construct each actual parser and stop at argument parsing, before any operation.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), patch.object(sys, 'argv', [str(path), '--help']):
        namespace = runpy.run_path(str(path), run_name='uap_help_probe')
        with patch.object(argparse.ArgumentParser, 'parse_args', capture):
            try:
                namespace['main']()
            except ParserCaptured:
                pass
    if len(captured) != 1:
        raise AssertionError(f'{path.name}: expected one CLI parser, found {len(captured)}')
    return captured[0]


def parser_problems(parser, script, commands=()):
    label = ' '.join((script, *commands))
    problems, count = [], 1
    description = parser.description or ''
    if not description.strip() or len(description.strip().splitlines()) != 1:
        problems.append(f'{label}: expected a one-line description')
    if not parser.epilog or 'Example:' not in parser.epilog or 'python ' not in parser.epilog:
        problems.append(f'{label}: missing runnable Python example')
    elif Path(script).name not in parser.epilog:
        problems.append(f'{label}: example names a different script')
    elif commands and ' '.join(commands) not in parser.epilog:
        problems.append(f'{label}: example omits its subcommand')
    else:
        example = shlex.split(parser.epilog.split('Example:', 1)[1].strip())
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                parser.parse_args(example[2 + len(commands):])
        except SystemExit:
            problems.append(f'{label}: example is rejected by its argument parser')
    rendered = parser.format_help()
    if script not in parser.prog:
        problems.append(f'{label}: rendered usage names a different script')
    if 'usage:' not in rendered or description not in rendered:
        # Wrapped descriptions are valid: compare whitespace rather than line breaks.
        if 'usage:' not in rendered or ' '.join(description.split()) not in ' '.join(rendered.split()):
            problems.append(f'{label}: description missing from rendered --help')
    for action in parser._actions:
        if action.help == argparse.SUPPRESS:
            continue
        if not action.help or not action.help.strip():
            problems.append(f'{label}: undocumented argument {action.dest}')
        if isinstance(action, argparse._SubParsersAction):
            documented = {choice.dest for choice in action._choices_actions if choice.help}
            for command, child in action.choices.items():
                if command not in documented:
                    problems.append(f'{label}: undocumented command {command}')
                child_problems, child_count = parser_problems(child, script, (*commands, command))
                problems.extend(child_problems)
                count += child_count
        else:
            for option in action.option_strings:
                if option not in rendered:
                    problems.append(f'{label}: option {option} missing from rendered --help')
            if (not action.option_strings and not action.choices
                    and action.dest.upper() not in rendered.upper()):
                problems.append(f'{label}: positional {action.dest} missing from rendered --help')
    return problems, count


def launcher_checks(repo):
    """Real --help must work without prompting, Python, a library or remembered paths."""
    bash = shutil.which('bash')
    if not bash and os.name == 'nt':
        candidate = Path(os.environ.get('ProgramFiles', '')) / 'Git/bin/bash.exe'
        bash = str(candidate) if candidate.is_file() else None
    names = ['Tag Library.bat', 'Sound Browser.bat', 'tag-library.sh', 'sound-browser.sh']
    tested = 0
    with tempfile.TemporaryDirectory(prefix='uap-help-') as temporary:
        root = Path(temporary)
        for name in names:
            path = root / name
            shutil.copy2(repo / name, path)
            env = dict(os.environ)
            if name.endswith('.bat'):
                if os.name != 'nt':
                    continue
                command = f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /s /c ""{path}" --help"'
            else:
                if not bash:
                    continue
                command = [bash, path.as_posix(), '--help']
                if os.name == 'nt':
                    env['PATH'] = str(Path(bash).resolve().parents[1] / 'usr/bin') + os.pathsep + env.get('PATH', '')
            def run():
                result = subprocess.run(command, cwd=root, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                                        encoding='utf-8', errors='replace', timeout=10)
                return result.returncode == 0 and all(text in result.stdout for text in
                        ('Usage:', 'Arguments:', 'Example:', 'library folder', 'demo', '--help'))
            assert run(), f'{name}: --help failed or omitted its description, arguments or example'
            assert not (root / 'library.txt').exists(), f'{name}: --help changed remembered library state'
            path.write_text(path.read_text(encoding='utf-8').replace('--help', '--broken-help'), encoding='utf-8')
            assert not run(), f'{name}: broken help routing escaped the runtime check'
            tested += 1
    print(f'PASS launcher help: {tested} runtime entry points; broken help routing rejected')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Example: python tools/help_tests.py', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--from', dest='source', type=Path,
                        help='explicit skill source (default: the public skill in this repository)')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    skill = args.source or repo / 'skills/audio-designer'
    for folder in [repo / 'tools', *(skill / name for name in ('scripts', 'bench', 'tagging', 'mix', 'render'))]:
        sys.path.insert(0, str(folder))
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    failures, total, parsers = [], 0, []
    for path in public_clis(repo, skill):
        captured = capture_parser(path)
        problems, count = parser_problems(captured, path.name)
        failures.extend(problems)
        total += count
        parsers.append(captured)
    # Deliberate regressions prove that the gate checks behavior, not just discovery.
    fixture = next(p for p in parsers if p.epilog and p.description)
    example = fixture.epilog
    fixture.epilog = None
    assert any('example' in issue for issue in parser_problems(fixture, 'fixture.py')[0]), 'missing example escaped'
    fixture.epilog = example
    description = fixture.description
    fixture.description = None
    assert any('description' in issue for issue in parser_problems(fixture, 'fixture.py')[0]), 'missing description escaped'
    fixture.description = description
    action = next(a for a in fixture._actions if a.option_strings)
    help_text = action.help
    action.help = None
    assert any('undocumented argument' in issue for issue in parser_problems(fixture, 'fixture.py')[0]), 'missing option help escaped'
    action.help = help_text
    for problem in failures:
        print('FAIL help: ' + problem)
    launcher_checks(repo)
    print(f'{"FAIL" if failures else "PASS"} CLI help: {len(parsers)} entry points, {total} command pages; '
          'missing example, description and argument mutations rejected')
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())

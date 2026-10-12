"""Calibration regression checks; every audio fixture is synthetic and temporary."""
import sys
sys.dont_write_bytecode = True

import copy
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import mix_check


def main():
    failures = []
    count = 0

    def check(name, condition):
        nonlocal count
        count += 1
        print(("PASS " if condition else "FAIL ") + name)
        if not condition:
            failures.append(name)

    def cli(*args):
        return subprocess.run([sys.executable, "-B", str(Path(mix_check.__file__)), *map(str, args)],
                              capture_output=True, text=True, encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="mix-calibration-") as temporary:
        root = Path(temporary)
        sr = 48000
        t = np.arange(8 * sr) / sr
        active = ((t >= 1) & (t < 3)) | ((t >= 4.5) & (t < 6.5))
        tone = np.sin(2 * np.pi * 220 * t)
        voice = tone * active
        voice *= 10 ** ((-16 - mix_check.audio_map.loudness_samples(voice, sr)["integrated_lufs"]) / 20)
        amplitude = np.max(np.abs(voice))

        def fixture(name, music_gap, bed_gap=None, sfx_gap=None):
            directory = root / name
            directory.mkdir()
            mix_check.write_test_wav(directory / "voice.wav", voice)
            mix_check.write_test_wav(directory / "music.wav", tone * amplitude * 10 ** (-music_gap / 20))
            if bed_gap is not None:
                mix_check.write_test_wav(directory / "bed.wav", tone * amplitude * 10 ** (-bed_gap / 20))
            if sfx_gap is not None:
                call = tone * amplitude * ((t >= 3.3) & (t < 3.7)) * 10 ** (-sfx_gap / 20)
                mix_check.write_test_wav(directory / "sfx.wav", call)
            return directory

        first, second = fixture("approved-a", 13, 38, 12), fixture("approved-b", 11, 36, 10)
        output = root / "levels.json"
        # Missing calibrate command, inverted gap direction, or bypassed metering fails this real CLI check.
        command = cli("calibrate", "--stems", first, second, "--out", output)
        check("calibrate CLI writes levels for two approved mixes", command.returncode == 0 and output.is_file())
        if not output.is_file():
            print(command.stderr.strip())
            return 1
        levels = json.loads(output.read_text(encoding="utf-8"))
        errors = []
        for measured, expected in zip(levels["calibration"]["mixes"], ((13, 38, 12), (11, 36, 10))):
            errors.extend((abs(measured["music_gap_lu"] - expected[0]),
                           abs(measured["bed_gap_lu"] - expected[1]),
                           abs(measured["sfx_peak_gap_db"][0] - expected[2])))
        print(f"METRIC known-gap maximum error: {max(errors):.4f} LU/dB")
        check("measured known music, bed and call gaps are within 0.3 LU/dB", max(errors) < .3)
        for rule, minimum, warning in (("music_gap", 9.5, 7.5), ("bed_gap", 34, 30), ("sfx_gap", 9, 6)):
            spec = levels["thresholds"][rule]
            check(rule + " applies its policy margin and tolerance to the lowest approved gap",
                  abs(spec["min"] - minimum) < .3 and abs(spec["warn_min"] - warning) < .3)

        candidate = fixture("candidate", 7)
        arguments = ("stems", "--voice", candidate / "voice.wav", "--music", candidate / "music.wav", "--json")
        original = copy.deepcopy(mix_check.THRESHOLDS)
        default_run = cli(*arguments)
        strict_run = cli(*arguments, "--levels", output)
        default_result = json.loads(default_run.stdout)
        strict_result = json.loads(strict_run.stdout)
        metric = lambda result: next(m for m in result["metrics"] if m["key"] == "music.gap_on")
        # Ignoring --levels, applying it only to the report, or mutating defaults fails here.
        check("7 LU candidate PASSes defaults and FAILs stricter calibrated music threshold",
              default_run.returncode == 0 and metric(default_result)["status"] == "PASS"
              and strict_run.returncode == 2 and metric(strict_result)["status"] == "FAIL")
        report = mix_check.markdown_report(strict_result)
        check("saved result reports the effective calibrated rule", "calibrated" in report.lower()
              and strict_result["thresholds"]["music_gap"]["min"] > 9)
        mix_check.inspect_mix(candidate / "voice.wav", candidate / "music.wav", levels=levels)
        check("using levels does not mutate the studio defaults", mix_check.THRESHOLDS == original)

        old_a, old_b = fixture("reference-a", 7.6, 35.5, 9), fixture("reference-b", 7.4, 31.9, 7.2)
        historic = mix_check.calibrate([old_a, old_b])
        deltas = {rule: {key: historic["thresholds"][rule][key] - mix_check.THRESHOLDS[rule][key]
                        for key in ("min", "warn_min")} for rule in ("music_gap", "bed_gap", "sfx_gap")}
        print("METRIC historical calibration minus defaults: " + json.dumps(deltas, sort_keys=True))
        check("approved historic gaps calibrate within 0.5 LU/dB of current thresholds",
              max(abs(value) for rule in deltas.values() for value in rule.values()) <= .5)
        check("calibration preserves compressor defaults that baked stems cannot identify",
              historic["ducking"] == {"music": {"threshold_db_re_voice": -3.8, "attack_ms": 20., "release_ms": 400.},
                                      "bed": {"threshold_db_re_voice": -13.3, "attack_ms": 10., "release_ms": 350.}})

        music_only = mix_check.calibrate([candidate])
        check("absent optional bed and calls keep their default thresholds",
              music_only["thresholds"]["bed_gap"] == original["bed_gap"]
              and music_only["thresholds"]["sfx_gap"] == original["sfx_gap"])
        malformed = copy.deepcopy(levels)
        malformed["thresholds"]["music_gap"]["warn_min"] = 100
        invalid = root / "invalid.json"
        invalid.write_text(json.dumps(malformed), encoding="utf-8")
        invalid_run = cli(*arguments, "--levels", invalid)
        check("inverted calibrated boundaries are rejected before assessment", invalid_run.returncode == 1
              and "ERROR:" in invalid_run.stderr)
        for name, invalid_policy in (("boolean schema", {"schema": True}),
                                     ("out-of-range attack", {"schema": 1, "ducking": {"music": {"attack_ms": 2001}}}),
                                     ("out-of-range release", {"schema": 1, "ducking": {"bed": {"release_ms": .001}}}),
                                     ("unsupported spectral split", {"schema": 1, "thresholds": {"brightness": {"hz": 5000}}})):
            rejected_policy = False
            try:
                mix_check.validate_levels(invalid_policy)
            except ValueError:
                rejected_policy = True
            check("invalid levels rejects " + name, rejected_policy)
        for rule, cases in (("music_gap", ((6, "PASS"), (5.9, "WARN"), (4, "WARN"), (3.9, "FAIL"), (20, "PASS"), (20.1, "WARN"))),
                            ("bed_gap", ((30, "PASS"), (29.9, "WARN"), (26, "WARN"), (25.9, "FAIL"))),
                            ("sfx_gap", ((6, "PASS"), (5.9, "WARN"), (3, "WARN"), (2.9, "FAIL"), (14, "PASS"), (14.1, "WARN")))):
            check(rule + " preserves default PASS/WARN/FAIL boundaries",
                  all(mix_check.gap_status(value, rule) == status for value, status in cases))
        silent = fixture("silent", 10)
        mix_check.write_test_wav(silent / "voice.wav", np.zeros_like(voice))
        rejected = cli("calibrate", "--stems", silent, "--out", root / "silent.json")
        check("silent voice cannot become an approved calibration", rejected.returncode == 1
              and not (root / "silent.json").exists())

    print(f"{count - len(failures)}/{count} calibration tests passed; {len(failures)} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

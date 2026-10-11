"""Synthetic integration and in-memory renderer mutation checks; no library audio."""
import csv
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import types

import numpy as np


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def metric(result, key):
    return next(m for m in result["metrics"] if m["key"] == key)


def fixtures(renderer, root):
    sr = 48000
    t = np.arange(10 * sr) / sr
    active = ((t >= 1) & (t < 3)) | ((t >= 8) & (t < 10))
    voice = .2 * np.sin(2 * np.pi * 220 * t) * active
    voice *= 10 ** ((-16 - renderer.audio_map.loudness_samples(np.column_stack((voice, voice)), sr)["integrated_lufs"]) / 20)
    rng = np.random.default_rng(921)
    signal = rng.uniform(-.08, .08, 6 * sr)
    sources = {
        "voice café.wav": voice,
        "music.wav": .25 * np.sin(2 * np.pi * 330 * np.arange(12 * sr) / sr),
        "bed.wav": .08 * np.sin(2 * np.pi * 180 * np.arange(12 * sr) / sr),
        "extrait café, 日本.wav": signal,
    }
    for name, samples in sources.items():
        renderer.mix_check.write_test_wav(root / name, np.column_stack((samples, samples)))
    columns = ["layer", "file", "excerpt_start", "excerpt_end", "cue_in", "cue_out",
               "gain_db", "duck_ratio", "highpass_hz", "lowpass_hz", "fade_in", "fade_out"]
    rows = [
        ["voice", "voice café.wav", 0, 10, 0, 12, 0, "-", "-", "-", 0, 0],
        ["music", "music.wav", 0, 12, 0, 12, -8, 6, "-", "-", 0, 0],
        ["bed", "bed.wav", 0, 12, 0, 12, -36, 8, 140, 1200, .01, .02],
        ["sfx", "extrait café, 日本.wav", 2, 4, 5, 7, -12, "-", "-", "-", 0, 0],
    ]
    cue = root / "cue café.csv"
    with cue.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(columns)
        writer.writerows(rows)
    return cue, signal[2 * sr:4 * sr] * 10 ** (-12 / 20)


def read(renderer, out, layer):
    return renderer.mix_check.decode(out / (layer + ".wav"))[0]


def timing(renderer, out, result, expected):
    y = read(renderer, out, "sfx")
    starts = np.flatnonzero(np.max(np.abs(y), axis=1) > 1e-7)
    require(len(starts) > 0 and abs(int(starts[0]) - 240000) <= 1,
            "cue did not land at sample 240000 (5 s ±1 sample frame)")
    require(np.max(np.abs(y[:239999]), initial=0) == 0, "audio precedes the cue")


def excerpt(renderer, out, result, expected):
    got = read(renderer, out, "sfx")[240000:336000, 0]
    require(len(got) == len(expected), "excerpt length changed")
    correlation = float(np.corrcoef(got, expected)[0, 1])
    require(correlation > .99999, f"wrong excerpt: correlation {correlation:.6f}")


def gain(renderer, out, result, expected):
    got = read(renderer, out, "sfx")[240000:336000, 0]
    original = expected / 10 ** (-12 / 20)
    db = 20 * math.log10(float(np.sqrt(np.mean(got.astype(float) ** 2))) /
                        float(np.sqrt(np.mean(original.astype(float) ** 2))))
    require(abs(db + 12) <= .5, f"-12 dB gain became {db:.3f} dB")


def ducking(renderer, out, result, expected):
    m = metric(result["mix_check"], "music.duck")
    require(m["value"] is not None and m["value"] >= 2,
            f"mix_check did not confirm ducking (approved shorts dip about 3 LU): {m}")
    # Beds also have a sidechain, though the existing checker assesses only their gap.
    v = read(renderer, out, "voice")
    b = read(renderer, out, "bed")
    p, counts = renderer.mix_check.energy_frames(v, 48000)
    on = p > .001
    bp, _ = renderer.mix_check.energy_frames(b, 48000)
    reduction = renderer.mix_check.level(bp, counts, ~on) - renderer.mix_check.level(bp, counts, on)
    require(reduction > 3, f"bed ducking missing: {reduction:.2f} dB")


def loudness(renderer, out, result, expected):
    value = renderer.audio_map.loudness(out / "final.wav")["integrated_lufs"]
    require(abs(value + 16) <= 1, f"final is {value:.1f} LUFS, expected -16 ±1")


def lengths(renderer, out, result, expected):
    sizes = [len(read(renderer, out, layer)) for layer in ("voice", "music", "bed", "sfx", "final")]
    require(sizes == [576000] * 5, f"unaligned lengths: {sizes}, expected 576000 each")


def mutant(renderer, old, new, name):
    source = Path(renderer.__file__).read_text(encoding="utf-8")
    require(source.count(old) == 1, f"mutation {name} must change exactly one production site")
    module = types.ModuleType("_render_mutant_" + name)
    module.__file__ = renderer.__file__
    sys.modules[module.__name__] = module
    exec(compile(source.replace(old, new), renderer.__file__, "exec"), module.__dict__)
    return module


def run(renderer):
    cases = [
        ("ducking", ducking,
         'graph.append(duck_graph(label, key, output, cue.duck_ratio, layer, voice_lufs))',
         'graph.append(f"[{key}]anullsink;[{label}]anull[{output}]")'),
        ("cue at 5 s ±1 sample frame", timing,
         'f"adelay={cue.in_frame}S:all=1"', 'f"adelay=0S:all=1"'),
        ("excerpt 2-4 s correlation", excerpt,
         'f"atrim=start_sample={cue.start_frame}:end_sample={cue.end_frame}"',
         'f"atrim=start_sample=0:end_sample={cue.end_frame - cue.start_frame}"'),
        ("gain -12 ±0.5 dB", gain,
         'f"volume={cue.gain_db:.12g}dB"', 'f"volume=0dB"'),
        ("final -16 ±1 LUFS", loudness,
         '"integrated_lufs": -16.0', '"integrated_lufs": -10.0'),
        ("equal stem lengths", lengths,
         'return f"apad=whole_len={frames},atrim=end_sample={frames}"', 'return "anull"'),
    ]
    failures = []
    # Every generated byte is inside render/; the directory is removed on exit.
    with tempfile.TemporaryDirectory(prefix=".check-", dir=Path(__file__).parent) as temporary:
        root = Path(temporary)
        cue, expected = fixtures(renderer, root)
        base = root / "good"
        result = renderer.render_mix(cue, base)
        for index, (name, validate, old, new) in enumerate(cases):
            try:
                validate(renderer, base, result, expected)
                broken = mutant(renderer, old, new, str(index))
                destination = root / ("broken-" + str(index))
                bad_result = broken.render_mix(cue, destination)
                rejected = False
                try:
                    validate(renderer, destination, bad_result, expected)
                except AssertionError:
                    rejected = True
                require(rejected, "test passed on the deliberately broken renderer")
                print("PASS " + name + "; corresponding broken renderer rejected", flush=True)
            except Exception as error:
                failures.append(name)
                print("FAIL " + name + ": " + str(error), flush=True)
        try:
            require(result["exit_code"] == 0, "good synthetic mix has checker FAILs")
            require(not any(m["status"] == "FAIL" for m in result["mix_check"]["metrics"]),
                    "good synthetic mix has checker FAILs")
            # The voice source ends at 10 s: the sidechain must preserve the 10-12 s tail.
            tail = read(renderer, base, "music")[11 * 48000:]
            require(np.sqrt(np.mean(tail.astype(float) ** 2)) > .01, "sidechain tail disappeared")
            require((base / "render.json").is_file(), "manifest missing")
            # Music 12 dB too loud: the checker FAILs it, and the CLI and report must say so.
            loud = root / "cue loud.csv"
            loud.write_text(cue.read_text(encoding="utf-8-sig").replace("music.wav,0,12,0,12,-8,", "music.wav,0,12,0,12,4,"),
                            encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                exit_status = renderer.main(["render", str(loud), "--out", str(root / "loud")])
            require(exit_status == 2, f"CLI returned {exit_status} on a checker FAIL")
            require("[FAIL]" in (root / "loud" / "cue_report.md").read_text(encoding="utf-8"),
                    "checker FAIL missing from written report")
            print("PASS checker/report/exit status; padded sidechain tail survives", flush=True)
        except Exception as error:
            failures.append("integration")
            print("FAIL integration: " + str(error), flush=True)
    print(f"{7 - len(failures)}/7 checks passed; 6 renderer mutations exercised; listening pending.")
    return 1 if failures else 0

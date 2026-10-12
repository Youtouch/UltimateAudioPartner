"""Synthetic integration and in-memory renderer mutation checks; no library audio."""
import csv
import copy
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


def cue_features(renderer, root):
    """Assert actual stem samples for short loops and source-peak placement."""
    sr = renderer.SAMPLE_RATE
    ramp = np.linspace(.05, .2, round(.23 * sr))
    renderer.mix_check.write_test_wav(root / "short.wav", ramp)
    spot = np.zeros((sr, 2))
    spot[:, 0] = .01 * np.sin(2 * np.pi * 211 * np.arange(sr) / sr)
    spot[round(.6 * sr), 1] = -.7  # Largest absolute peak is negative and in the right channel.
    renderer.mix_check.write_test_wav(root / "spot.wav", spot)
    path = root / "features.csv"
    path.write_text(
        "# Commented cue-sheet template.\n"
        "layer,file,excerpt_start,excerpt_end,cue_in,cue_out,fade_in,fade_out,loop,align\n"
        "music,short.wav,0,.23,0,2,0,0,yes,start\n"
        "# A second looping layer and effects with and without a pre-zero onset.\n"
        "bed,short.wav,0,.23,0,2,0,0,yes,\n"
        "sfx,spot.wav,.2,.9,.75,,0,0,no,peak\n",
        encoding="utf-8")

    def render(module, csv_path, name, layers):
        _, cues = module.read_cues(csv_path)
        out = root / name
        out.mkdir()
        for layer in layers:
            module.render_layer(layer, cues, max(c.out_frame for c in cues), out, module.CommandLog())
        return out, cues

    def loop_samples(module, out):
        for layer in ("music", "bed"):
            got = read(module, out, layer)[:, 0]
            require(len(got) == 2 * sr, "loop changed cue length")
            require(np.min(got[round(.25 * sr):]) > .04,
                    "short source went silent before the looping cue ended")
            require(np.max(np.abs(np.diff(got))) < .001,
                    "loop seam was cut without its short crossfade")

    out, cues = render(renderer, path, "features", ("music", "bed", "sfx"))
    loop_samples(renderer, out)
    # The exact source peak, not the excerpt start or a positive-only/mono peak, lands on the cue.
    got = read(renderer, out, "sfx")
    require(int(np.argmax(np.max(np.abs(got), axis=1))) == round(.75 * sr),
            "sfx peak missed its cue time")
    require(cues[-1].out_frame == round(1.05 * sr),
            "omitted peak-aligned cue_out did not preserve the shifted excerpt end")

    no_loop = mutant(renderer, 'if not cue.loop or length >= cue.play_frames:',
                     'if True:', 'no_loop')
    broken, _ = render(no_loop, path, "features-no-loop", ("music", "bed"))
    try:
        loop_samples(renderer, broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("loop regression accepted silence after a short source")
    no_crossfade = mutant(renderer,
                         'f"[{left}][{right}]acrossfade=ns={crossfade}:c1=tri:c2=tri[{label}]"',
                         'f"[{left}][{right}]concat=n=2:v=0:a=1[{label}]"', 'no_crossfade')
    broken, _ = render(no_crossfade, path, "features-cut-loop", ("music", "bed"))
    try:
        loop_samples(renderer, broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("loop regression accepted an abrupt seam")

    no_peak = mutant(renderer,
                     'peak = source_peak_frame(file, channels, a, b) if align == "peak" else 0',
                     'peak = 0', 'no_peak')
    broken, _ = render(no_peak, path, "features-no-peak", ("sfx",))
    broken_peak = int(np.argmax(np.max(np.abs(read(renderer, broken, "sfx")), axis=1)))
    require(broken_peak != round(.75 * sr), "peak regression accepted start alignment")

    zero = root / "peak-zero.csv"
    zero.write_text(path.read_text(encoding="utf-8").replace(".9,.75,,", ".9,0,,"), encoding="utf-8")
    out, _ = render(renderer, zero, "features-zero", ("sfx",))
    got = read(renderer, out, "sfx")
    require(int(np.argmax(np.max(np.abs(got), axis=1))) == 0,
            "a pre-zero onset moved the peak away from time zero")
    require(abs(float(got[0, 1]) + .7) < 1e-6 and np.max(np.abs(got[round(.3 * sr):]), initial=0) == 0,
            "clipping a pre-zero onset changed peak amplitude or the tail end")

    default = root / "default.csv"
    default.write_text("layer,file,cue_in,cue_out,fade_in,fade_out\nmusic,short.wav,0,2,0,0\n", encoding="utf-8")
    out, _ = render(renderer, default, "features-default", ("music",))
    require(np.max(np.abs(read(renderer, out, "music")[sr:]), initial=0) == 0,
            "omitting loop changed the original silence-padding behavior")
    for row in ("voice,short.wav,yes,start", "sfx,spot.wav,yes,peak",
                "music,short.wav,no,peak", "bed,short.wav,maybe,start", "sfx,spot.wav,no,centre"):
        invalid = root / "invalid.csv"
        invalid.write_text("layer,file,loop,align\n" + row + "\n", encoding="utf-8")
        try:
            renderer.read_cues(invalid)
        except ValueError:
            continue
        raise AssertionError("invalid loop/align combination was accepted: " + row)


def calibrated_levels(renderer, root, cue, base, result):
    require(renderer.DUCK == renderer.mix_check.DEFAULT_DUCK, "renderer/checker duck defaults differ")
    policy = {"schema": 1, "thresholds": copy.deepcopy(renderer.mix_check.THRESHOLDS),
              "ducking": copy.deepcopy(renderer.DUCK)}
    policy["thresholds"]["music_gap"].update(min=60., warn_min=58., max=80.)
    policy["ducking"]["music"]["threshold_db_re_voice"] -= 18.
    path = root / "levels.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    out = root / "strict"
    with redirect_stdout(io.StringIO()):
        status = renderer.main(["render", str(cue), "--out", str(out), "--levels", str(path)])
    require(status == 2, "renderer ignored calibrated checker thresholds")
    manifest = json.loads((out / "render.json").read_text(encoding="utf-8"))
    require(metric(manifest["mix_check"], "music.gap_on")["status"] == "FAIL",
            "strict calibrated music gap was not applied")
    require(manifest["levels"]["thresholds"]["music_gap"]["min"] == 60.,
            "render manifest omitted the effective calibrated policy")
    require("[FAIL]" in (out / "cue_report.md").read_text(encoding="utf-8"),
            "calibrated FAIL was missing from the report")

    # The file's lower threshold must reduce real samples through the render CLI.
    normal = read(renderer, base, "music")[2 * 48000:3 * 48000]
    changed = read(renderer, out, "music")[2 * 48000:3 * 48000]
    ratio = np.sqrt(np.mean(changed.astype(float) ** 2)) / np.sqrt(np.mean(normal.astype(float) ** 2))
    require(ratio < .4, f"custom ducking did not affect rendered music: amplitude ratio {ratio:.3f}")
    try:
        renderer.duck_graph("music", "voice", "out", 3, "music", -80, policy["ducking"])
    except ValueError as error:
        require("threshold" in str(error), "extreme ducking setting lacked an actionable error")
    else:
        raise AssertionError("extreme ducking setting reached FFmpeg outside its accepted threshold range")


def run(renderer):
    cases = [
        ("ducking", ducking,
         'graph.append(duck_graph(label, key, output, cue.duck_ratio, layer, voice_lufs, ducking))',
         'graph.append(f"[{key}]anullsink;[{label}]anull[{output}]")'),
        ("cue at 5 s ±1 sample frame", timing,
         'f"adelay={delay}S:all=1"', 'f"adelay=0S:all=1"'),
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
    with tempfile.TemporaryDirectory(prefix="render-check-") as temporary:
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
        for name, check in (("loop/crossfade/peak/defaults/input validation",
                             lambda: cue_features(renderer, root)),
                            ("calibrated checker and ducking",
                             lambda: calibrated_levels(renderer, root, cue, base, result))):
            try:
                check()
                print("PASS " + name, flush=True)
            except Exception as error:
                failures.append(name)
                print("FAIL " + name + ": " + str(error), flush=True)
    print(f"{9 - len(failures)}/9 checks passed; 9 renderer mutations exercised; listening pending.")
    return 1 if failures else 0

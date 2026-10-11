"""CPU-only mix checks against references/mix-plan.md; no recognition models."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
import types
import wave

import numpy as np

# All assessment and gate thresholds live here. "policy" values operationalise
# approximate/qualitative plan wording; they are NOT extra rules from the plan.
THRESHOLDS = {
    "measurement": {"decimals": 1, "source": "mix-plan.md:15,48",
                    "policy": "assess gaps/duck at displayed 0.1 dB precision; avoid false boundary WARNs"},
    "voice_lufs": {"target": -16.0, "pass_delta": 1.0, "warn_delta": 2.0,
                   "source": "mix-plan.md:9,15", "policy": "1/2 LU tolerances"},
    "music_gap": {"min": 6.0, "max": 20.0, "warn_min": 4.0,
                  "source": "mix-plan.md:12,15",
                  "policy": "two approved narrated shorts measure 7.6/7.4 LU here; about 1.5 LU margin, 2 LU tolerance"},
    "duck": {"source": "mix-plan.md:12,27",
             "policy": "reported, not judged: the gap under speech is the requirement, ducking one way to hold it"},
    "bed_gap": {"min": 30.0, "warn_min": 26.0,
                "source": "mix-plan.md:13,15,30",
                "policy": "two approved beds measure 35.5/31.9 LU under speech here; 4 LU tolerance"},
    "brightness": {"hz": 3500.0, "fraction": 0.5,
                   "source": "mix-plan.md:30",
                   "policy": "reported: K-weighting already lifts the highs; an approved cricket bed was 77% bright"},
    "bed_lra": {"warn_min": 18.0, "source": "mix-plan.md:31",
                "policy": "WARN at or above the cited near-18 LU example"},
    "stem_peak": {"pass_max": -1.5, "fail_min": 0.0,
                  "source": "mix-plan.md:43,44", "policy": "reuse final headroom as stem WARN boundary; >=0 fails"},
    "mix_lufs": {"target": -16.0, "pass_delta": 1.0, "warn_delta": 2.0,
                 "source": "mix-plan.md:15,43", "policy": "1/2 LU tolerances"},
    "mix_peak": {"pass_max": -1.5, "warn_max": -1.0,
                 "source": "mix-plan.md:43", "policy": "0.5 dB overshoot tolerance"},
    "sfx_gap": {"min": 6.0, "max": 14.0, "warn_min": 3.0,
                "source": "mix-plan.md:11,15",
                "policy": "calls in pauses: an approved owl call 9.0, a reference monkey call 7.2; 1 dB margin"},
    "sfx_overlap": {"max": 0.1, "source": "mix-plan.md:36",
                    "policy": "effect activity in a voice window fails beyond one 0.1 s analysis frame (gate edges)"},
    "coverage": {"tolerance_s": 0.1, "source": "mix-plan.md:28,48",
                 "policy": "allow one analysis frame of duration rounding; missing coverage fails"},
    "gate": {"frame_s": 0.1, "relative_db": 18.0, "floor_dbfs": -55.0,
             "percentile": 95.0, "minimum_s": 0.4,
             "source": "mix-plan.md:48,50,51",
             "policy": "local energy heuristic, NOT speech recognition; frame/floor/depth not in plan"},
    "ebu_floor": {"lufs": -70.0, "source": "mix-plan.md:48",
                  "policy": "EBU absolute gate; -70 sentinel is unavailable, not a measured quiet level"},
}

# Import only the existing numpy/stdlib helpers, without writing into scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audio_map

SAMPLE_RATE = 48000  # Analysis resolution, not an assessment threshold.


def decode(path):
    """Route the shared loader through its existing FFmpeg fallback.

    audio_map.load_audio imports soundfile unconditionally. A temporary, minimal
    adapter raises the ValueError that already selects its FFmpeg branch. This
    avoids a soundfile dependency and does not duplicate or edit its decoder.
    This CLI is single-threaded; restore sys.modules immediately afterwards.
    """
    adapter = types.ModuleType("soundfile")

    def ffmpeg_only(*args, **kwargs):
        raise ValueError("Use audio_map's existing FFmpeg decoder")

    adapter.SoundFile = ffmpeg_only
    adapter.LibsndfileError = ValueError
    previous = sys.modules.get("soundfile")
    sys.modules["soundfile"] = adapter
    try:
        return audio_map.load_audio(path, sample_rate=SAMPLE_RATE)
    finally:
        if previous is None:
            sys.modules.pop("soundfile", None)
        else:
            sys.modules["soundfile"] = previous


def k_weight(y, sr):
    """ITU-R BS.1770 K-weighting (the LUFS filter), so frame levels and gaps are in the plan's unit."""
    from scipy.signal import lfilter
    assert sr == 48000, "K-weighting coefficients are for 48 kHz"
    y = lfilter([1.53512485958697, -2.69169618940638, 1.19839281085285],
                [1.0, -1.69065929318241, 0.73248077421585], y, axis=0)
    return lfilter([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621], y, axis=0)


def energy_frames(y, sr):
    """K-weighted mean channel power, never a mono sum (antiphase stereo must survive)."""
    if y.ndim == 1:
        y = y[:, None]
    y = k_weight(y.astype(np.float64), sr)
    size = round(sr * THRESHOLDS["gate"]["frame_s"])
    full, tail = divmod(len(y), size)
    power = np.mean(y[:full * size].reshape(full, size, y.shape[1]).astype(np.float64) ** 2,
                    axis=(1, 2)) if full else np.empty(0)
    counts = np.full(full, size, dtype=np.int64)
    if tail:
        power = np.append(power, np.mean(y[-tail:].astype(np.float64) ** 2))
        counts = np.append(counts, tail)
    return power, counts


def level(power, counts, mask):
    if not np.any(mask):
        return None
    mean = float(np.average(power[mask], weights=counts[mask]))
    return 10 * math.log10(mean) if mean > 0 else None


def target_status(value, rule):
    spec = THRESHOLDS[rule]
    delta = abs(value - spec["target"])
    return "PASS" if delta <= spec["pass_delta"] else "WARN" if delta <= spec["warn_delta"] else "FAIL"


def gap_status(value, rule, bright=False):
    value = round(value, THRESHOLDS["measurement"]["decimals"])
    spec = THRESHOLDS[rule]
    extra = 0
    if value < spec["warn_min"] + extra:
        return "FAIL"
    if value < spec["min"] + extra or value > spec.get("max", math.inf):
        return "WARN"
    return "PASS"


def inspect_mix(voice, music, bed=None, sfx=(), mix=None):
    result = {"schema": 1, "rows": [], "metrics": [], "notes": [
        "Measured evidence only; listening pending. No models, tags or network calls.",
        "Inputs must be edited, aligned at time zero, and have their gains/ducking baked in.",
        "Gaps are K-weighted (BS.1770, the LUFS filter) levels in gated speech windows, in LU; not differences of integrated LUFS.",
        "Outside-speech music gap uses the ACTIVE voice RMS reference; silence is not a voice reference.",
        "Ducking is music RMS outside minus inside the gate; arrangement changes can mimic it.",
        "The energy gate is not a voice detector. EQ, identity, coherence and masking remain unverified.",
        "INFO rows are reference measurements with no plan target. PASS/WARN/FAIL use approximate starting levels plus explicit local tolerances; see THRESHOLDS.",
    ]}

    def add(row, key, value, unit, status, rule, detail):
        if value is None or not math.isfinite(value):
            value = None
            if status not in ("FAIL", "INFO"):
                status = "WARN"
            detail = "Unavailable (silence, EBU gate, or no comparable windows). " + detail
        item = {"key": key, "value": value, "unit": unit, "status": status,
                "source": THRESHOLDS[rule]["source"], "detail": detail}
        result["metrics"].append(item)
        row["metrics"].append(item)
        return item

    tracks = {}

    def track(role, key, path):
        path = Path(path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Not a file: " + str(path))
        y, sr = decode(path)
        if not len(y):
            raise ValueError("No decoded audio: " + str(path))
        if not np.all(np.isfinite(y)):
            raise ValueError("Non-finite audio samples: " + str(path))
        power, counts = energy_frames(y, sr)
        row = {"layer": role, "file": str(path), "duration_s": len(y) / sr,
               "duck_eq": "unknown; measure supplied audio", "metrics": []}
        result["rows"].append(row)
        tracks[key] = {"row": row, "power": power, "counts": counts, "sr": sr}
        meters = audio_map.loudness(path)  # whole file, ebur128=peak=true
        integrated = meters["integrated_lufs"]
        if integrated <= THRESHOLDS["ebu_floor"]["lufs"]:
            integrated = None
        rule = "mix_lufs" if key == "mix" else "voice_lufs"
        status = target_status(integrated, rule) if integrated is not None and key in {"voice", "mix"} else (
            "WARN" if key in {"voice", "mix"} else "INFO")
        detail = ("Target about -16 LUFS; PASS within 1 LU, WARN within 2 LU."
                  if key in {"voice", "mix"} else
                  "No absolute LUFS target for this stem; judge its voice-relative gap/peak.")
        add(row, key + ".integrated", integrated, "LUFS", status, rule if key in {"voice", "mix"} else "ebu_floor", detail)
        peak = meters["true_peak_dbtp"]
        if key == "mix":
            spec = THRESHOLDS["mix_peak"]
            status = "PASS" if peak <= spec["pass_max"] else "WARN" if peak <= spec["warn_max"] else "FAIL"
            rule = "mix_peak"
            detail = "Ceiling -1.5 dBTP; WARN up to -1.0, FAIL above."
        else:
            spec = THRESHOLDS["stem_peak"]
            status = "FAIL" if peak >= spec["fail_min"] else "PASS" if peak <= spec["pass_max"] else "WARN"
            rule = "stem_peak"
            detail = "No clipping: FAIL at/above 0 dBTP; WARN above -1.5 headroom."
        add(row, key + ".peak", peak, "dBTP", status, rule, detail)
        tracks[key]["peak"] = peak
        if key == "bed":
            # The shared bands split at exactly 3500 Hz; sum the two upper bands.
            fraction = sum(audio_map.band_shares(y, sr)[3:]) if np.any(power > 0) else None
            bright = fraction is not None and fraction > THRESHOLDS["brightness"]["fraction"]
            tracks[key]["bright"] = bright
            add(row, "bed.brightness", fraction * 100 if fraction is not None else None, "%",
                "INFO", "brightness",
                "Energy above 3.5 kHz (unweighted); the K-weighted gap already counts the highs louder.")
            lra = meters["loudness_range_lu"]
            add(row, "bed.lra", lra if integrated is not None else None, "LU",
                "WARN" if lra >= THRESHOLDS["bed_lra"]["warn_min"] else "PASS", "bed_lra",
                "Bursts: WARN at/above 18 LU LRA; short clips can understate range.")
        del y
        return tracks[key]

    v = track("Voice", "voice", voice)
    p, c = v["power"], v["counts"]
    gate = THRESHOLDS["gate"]
    positive = p[p > 0]
    gate_db = max(gate["floor_dbfs"], 10 * math.log10(float(np.percentile(positive, gate["percentile"])))
                  - gate["relative_db"]) if len(positive) else gate["floor_dbfs"]
    active = p > 10 ** (gate_db / 10)
    active_s = float(c[active].sum()) / v["sr"]
    pause_s = float(c[~active].sum()) / v["sr"]
    v["row"]["duck_eq"] = "short-window energy gate; EQ unknown"
    add(v["row"], "voice.gate", active_s, "s", "PASS" if active_s >= gate["minimum_s"] else "FAIL",
        "gate", "Gated activity, not recognised speech; need at least 0.4 s.")
    add(v["row"], "voice.pauses", pause_s, "s", "PASS" if pause_s >= gate["minimum_s"] else "WARN",
        "gate", "Need at least 0.4 s of pauses to assess ducking.")
    add(v["row"], "voice.gate_threshold", gate_db, "dBFS", "INFO", "gate",
        "Heuristic reference only: max(-55 dBFS, 95th-percentile frame level minus 18 dB).")
    add(v["row"], "voice.active_level", level(p, c, active), "dB K-weighted", "INFO", "gate",
        "Active voice reference; no absolute RMS target in the plan.")

    def compare(key, t):
        row = t["row"]
        n = min(len(p), len(t["power"]))
        weights = np.minimum(c[:n], t["counts"][:n])
        on = active[:n]
        off = ~on
        voiced = level(p[:n], weights, on)
        on_level = level(t["power"][:n], weights, on)
        off_level = level(t["power"][:n], weights, off)
        gap_on = voiced - on_level if voiced is not None and on_level is not None else None
        tolerance = THRESHOLDS["coverage"]["tolerance_s"]
        delta = row["duration_s"] - v["row"]["duration_s"]
        coverage = float(weights.sum()) / float(c.sum()) * 100
        add(row, key + ".coverage", coverage, "%", "FAIL" if delta < -tolerance else "WARN" if delta > tolerance else "PASS",
            "coverage", "Continuous layer should cover the voice timeline; excess tail is WARN; no implicit padding/looping.")
        add(row, key + ".level_on", on_level, "dB K-weighted", "INFO", "gate",
            "Level in shared voice-active windows; reference measurement, no absolute plan target.")
        rule = "music_gap" if key == "music" else "bed_gap"
        bright = key == "bed" and t["bright"]
        add(row, key + ".gap_on", gap_on, "LU",
            gap_status(gap_on, rule, bright) if gap_on is not None else "WARN", rule,
            "Voice minus layer during gated speech. " + (
                "PASS 6-20 LU; WARN 4-6 or >20; FAIL <4." if key == "music" else
                "PASS >=30 LU; WARN 26-30; FAIL <26."))
        if key == "music":
            row["duck_eq"] = "outside-minus-inside level difference; EQ unknown"
            add(row, "music.level_off", off_level, "dB K-weighted", "INFO", "gate",
                "Music level in voice pauses, within the common timeline; no absolute target.")
            gap_off = voiced - off_level if voiced is not None and off_level is not None else None
            add(row, "music.gap_off", gap_off, "LU", "INFO", "music_gap",
                "ACTIVE voice reference minus pause music; plan has no outside-speech target.")
            duck = off_level - on_level if (off_level is not None and on_level is not None
                    and pause_s >= gate["minimum_s"] and active_s >= gate["minimum_s"]
                    and float(weights[off].sum()) / v["sr"] >= gate["minimum_s"]) else None
            if duck is not None:
                duck = round(duck, THRESHOLDS["measurement"]["decimals"])
            add(row, "music.duck", duck, "LU", "INFO", "duck",
                "Music level outside speech minus inside. Reported only; natural dynamics can mimic ducking.")
        else:
            row["duck_eq"] = "bright-bed allowance" if bright else "dark-bed gap; EQ unknown"

    compare("music", track("Music", "music", music))
    if bed is not None:
        compare("bed", track("Bed", "bed", bed))
    for index, path in enumerate(sfx, 1):
        key = "sfx" + str(index)
        t = track("Effect " + str(index), key, path)
        n = min(len(p), len(t["power"]))
        weights = np.minimum(c[:n], t["counts"][:n])
        peak_gap = v["peak"] - t["peak"] if math.isfinite(v["peak"]) and math.isfinite(t["peak"]) else None
        add(t["row"], key + ".peak_gap", peak_gap, "dB",
            gap_status(peak_gap, "sfx_gap") if peak_gap is not None else "WARN", "sfx_gap",
            "Voice true peak minus effect true peak; PASS 6-14 dB; WARN 3-6 or >14; FAIL <3.")
        positive = t["power"][t["power"] > 0]
        threshold = max(gate["floor_dbfs"], 10 * math.log10(float(np.percentile(positive, gate["percentile"])))
                        - gate["relative_db"]) if len(positive) else gate["floor_dbfs"]
        activity = t["power"][:n] > 10 ** (threshold / 10)
        overlap = float(weights[activity & active[:n]].sum()) / v["sr"]
        add(t["row"], key + ".overlap", overlap, "s",
            "FAIL" if overlap > THRESHOLDS["sfx_overlap"]["max"] + 1e-9 else "PASS", "sfx_overlap",
            "Effect activity under voice fails beyond one 0.1 s frame (the gate's edge).")
        t["row"]["duck_eq"] = "effect/spot-call pause rule; game/UI intent unknown"
    if mix is not None:
        track("Final mix", "mix", mix)
    else:
        result["notes"].append("WARN: no --mix supplied; final LUFS/true-peak assessment pending.")
    return result


def format_metric(metric, explanation=False):
    value = "n/a" if metric["value"] is None else f'{metric["value"]:.1f}'
    label = metric["key"].split(".", 1)[-1]
    text = f'{label}: {value} {metric["unit"]} [{metric["status"]}]'
    if explanation:
        text += f' — {metric["detail"]} ({metric["source"]})'
    return text


def markdown_report(result):
    def cell(value):
        return str(value).replace("|", r"\|").replace("\r", " ").replace("\n", " ")

    lines = ["| Layer | File | Excerpt | Cue in/out | Gain | Duck / EQ | Measured after gain | Listening |",
             "|---|---|---|---|---|---|---|---|"]
    for row in result["rows"]:
        values = [row["layer"], row["file"], "whole supplied stem",
                  f'0.0-{row["duration_s"]:.2f} s', "baked in; unknown",
                  row["duck_eq"], "; ".join(format_metric(m) for m in row["metrics"]), "pending"]
        lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    lines += ["", *["- " + note for note in result["notes"]], "",
              "Assessment rules (numeric tolerances are documented local policies):", ""]
    for name, spec in THRESHOLDS.items():
        lines.append(f'- {name}: {spec["source"]}; {spec["policy"]}.')
    return "\n".join(lines)


def print_stems(result):
    for row in result["rows"]:
        print(f'{row["layer"]}: {row["file"]}')
        for metric in row["metrics"]:
            print("  " + format_metric(metric, explanation=True))
    print()
    for note in result["notes"]:
        print(note)


def write_test_wav(path, signal, sr=SAMPLE_RATE):
    """Test fixture only: 32-bit PCM avoids quantising very quiet beds away."""
    signal = np.asarray(signal)
    if signal.ndim == 1:
        signal = signal[:, None]
    if np.max(np.abs(signal), initial=0) >= 1:
        raise ValueError("Test fixture would clip")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(signal.shape[1])
        output.setsampwidth(4)
        output.setframerate(sr)
        output.writeframes(np.round(signal * (2 ** 31 - 1)).astype("<i4").tobytes())


def selftest():
    """Each rejection assertion catches a plausible false PASS on broken audio."""
    failures = []
    count = 0

    def test(name, predicate):
        nonlocal count
        count += 1
        try:
            assert predicate(), name
            print("PASS " + name)
        except (AssertionError, KeyError, StopIteration) as error:
            failures.append(name)
            print("FAIL " + name + ": " + str(error))

    def metric(result, key):
        return next(m for m in result["metrics"] if m["key"] == key)

    with tempfile.TemporaryDirectory(prefix="mix-check-") as directory:
        root = Path(directory)
        sr = SAMPLE_RATE
        t = np.arange(8 * sr) / sr
        active = ((t >= 1) & (t < 3)) | ((t >= 4.5) & (t < 6.5))
        rng = np.random.default_rng(20261011)
        voice = (np.sin(2 * np.pi * 220 * t) + .08 * rng.standard_normal(len(t))) * active
        voice *= 10 ** ((-16 - audio_map.loudness_samples(voice, sr)["integrated_lufs"]) / 20)
        ref = float(np.sqrt(np.mean(k_weight(voice, sr)[active] ** 2)))  # K-weighted, as the checker measures

        def at_rms(y, rms):
            return y * (rms / np.sqrt(np.mean(y ** 2)))

        def at_k(y, rms):
            return y * (rms / np.sqrt(np.mean(k_weight(y, sr) ** 2)))

        music_on = at_k(np.sin(2 * np.pi * 500 * t), ref * 10 ** (-18 / 20))
        ducked = music_on * np.where(active, 1.0, 10 ** (6 / 20))
        frequencies = np.fft.rfftfreq(len(t), 1 / sr)
        noise = np.fft.rfft(rng.standard_normal(len(t)))
        dark = np.fft.irfft(noise * (frequencies < 1200), n=len(t))
        bright = np.fft.irfft(noise * ((frequencies > 5000) & (frequencies < 9000)), n=len(t))
        dark = at_k(dark, ref * 10 ** (-34 / 20))
        bright = at_k(bright, ref * 10 ** (-34 / 20))
        bright_raw = at_rms(bright, np.sqrt(np.mean(dark ** 2)))  # same unweighted level as the dark bed
        master = voice + ducked + dark
        master *= 10 ** ((-16 - audio_map.loudness_samples(master, sr)["integrated_lufs"]) / 20)
        signals = {
            "voice café.wav": voice, "music ducked.wav": ducked,
            "music unducked.wav": music_on, "bed dark.wav": dark,
            "bed loud.wav": dark * 10 ** (10 / 20), "bed bright.wav": bright,
            "bed bright raw.wav": bright_raw, "music close.wav": music_on * 10 ** (15 / 20),
            "master.wav": master, "master loud.wav": master * 10 ** (6 / 20),
            "voice silent.wav": np.zeros_like(voice),
            "voice continuous.wav": at_rms(np.sin(2 * np.pi * 220 * t), ref),
            "music short.wav": ducked[:2 * sr],
            "voice antiphase.wav": np.column_stack((voice, -voice)),
            "music antiphase.wav": np.column_stack((ducked, -ducked)),
            "effect pause.wav": at_rms(np.sin(2 * np.pi * 700 * t), ref) * ((t > 3.3) & (t < 3.7)) * .25,
            "effect overlap.wav": at_rms(np.sin(2 * np.pi * 700 * t), ref) * ((t > 1.3) & (t < 1.7)) * .25,
        }
        for name, signal in signals.items():
            write_test_wav(root / name, signal)

        def run(music="music ducked.wav", bed="bed dark.wav", mix="master.wav",
                voice="voice café.wav", sfx=()):
            return inspect_mix(root / voice, root / music, root / bed if bed else None,
                               [root / p for p in sfx], root / mix if mix else None)

        good = run()
        test("known 18 dB music gap is measured and passes",
             lambda: abs(metric(good, "music.gap_on")["value"] - 18) < .3
             and metric(good, "music.gap_on")["status"] == "PASS")
        test("known 6 dB duck is measured and reported (INFO, not judged)",
             lambda: abs(metric(good, "music.duck")["value"] - 6) < .3
             and metric(good, "music.duck")["status"] == "INFO")
        test("known dark bed gap and master pass",
             lambda: metric(good, "bed.gap_on")["status"] == "PASS"
             and metric(good, "mix.integrated")["status"] == "PASS")
        test("a mix built to the plan raises no WARN or FAIL (reference rows are INFO)",
             lambda: not [m["key"] for m in good["metrics"] if m["status"] in ("WARN", "FAIL")])
        unducked = run(music="music unducked.wav")
        test("unducked music at a good gap under speech passes (the gap is the requirement)",
             lambda: metric(unducked, "music.gap_on")["status"] == "PASS"
             and abs(metric(unducked, "music.duck")["value"]) < .3)
        wrong = run(music="music close.wav")
        test("BROKEN music 3 LU under the voice FAILS",
             lambda: metric(wrong, "music.gap_on")["status"] == "FAIL")
        wrong = run(bed="bed loud.wav")
        test("BROKEN bed 10 dB too loud FAILS",
             lambda: metric(wrong, "bed.gap_on")["status"] == "FAIL")
        wrong = run(mix="master loud.wav")
        test("BROKEN -10 LUFS master FAILS",
             lambda: abs(metric(wrong, "mix.integrated")["value"] + 10) < .3
             and metric(wrong, "mix.integrated")["status"] == "FAIL")
        bright_result = run(bed="bed bright.wav")
        test("brightness is reported: bright noise > 90%, dark < 1%",
             lambda: metric(bright_result, "bed.brightness")["value"] > 90
             and metric(good, "bed.brightness")["value"] < 1)
        raw = run(bed="bed bright raw.wav")
        test("K-weighting: bright noise at the dark bed's unweighted level measures at least 3 LU closer to the voice",
             lambda: metric(good, "bed.gap_on")["value"] - metric(raw, "bed.gap_on")["value"] >= 3)
        wrong = run(voice="voice silent.wav")
        test("BROKEN silent voice FAILS the gate",
             lambda: metric(wrong, "voice.gate")["status"] == "FAIL")
        continuous = run(voice="voice continuous.wav")
        test("no pauses: ducking is unavailable, not invented",
             lambda: metric(continuous, "music.duck")["value"] is None)
        wrong = run(music="music short.wav")
        test("BROKEN truncated music FAILS coverage instead of padding silence",
             lambda: metric(wrong, "music.coverage")["status"] == "FAIL")
        stereo = run(voice="voice antiphase.wav", music="music antiphase.wav")
        test("antiphase stereo does not cancel the gap measurement",
             lambda: abs(metric(stereo, "music.gap_on")["value"] - 18) < .3)
        pause = run(sfx=["effect pause.wav"])
        overlap = run(sfx=["effect overlap.wav"])
        test("effect in a pause passes; BROKEN effect under voice FAILS",
             lambda: metric(pause, "sfx1.overlap")["status"] == "PASS"
             and metric(overlap, "sfx1.overlap")["status"] == "FAIL")
        for input_name, expected in (("music ducked.wav", 0), ("music close.wav", 2)):
            result = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "stems", "--voice",
                 str(root / "voice café.wav"), "--music", str(root / input_name)],
                capture_output=True, text=True, encoding="utf-8")
            test("CLI exit status for " + input_name, lambda r=result, e=expected: r.returncode == e)
        text_result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "report", "--voice",
             str(root / "voice café.wav"), "--music", str(root / "music ducked.wav")],
            capture_output=True, text=True, encoding="utf-8")
        test("report is a cue-sheet table with measurements and listening pending",
             lambda: text_result.returncode == 0
             and "| Layer | File | Excerpt | Cue in/out | Gain | Duck / EQ | Measured after gain | Listening |" in text_result.stdout
             and "18.0 LU [PASS]" in text_result.stdout and "pending" in text_result.stdout)

    print(f"{count - len(failures)}/{count} tests passed; {len(failures)} failed.")
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="synthetic CPU-only positive and broken-input self-tests")
    for name in ("stems", "report"):
        command = commands.add_parser(name, help="measure aligned stems" if name == "stems" else "cue-sheet Markdown")
        command.add_argument("--voice", type=Path)
        command.add_argument("--music", type=Path)
        command.add_argument("--bed", type=Path)
        command.add_argument("--sfx", type=Path, nargs="+", action="extend", default=[])
        command.add_argument("--mix", type=Path)
        if name == "stems":
            command.add_argument("--json", action="store_true", help="emit measurements as JSON to stdout")
        else:
            command.add_argument("--input", help="render prior stems --json output; '-' reads stdin")
    args = parser.parse_args(argv)
    if args.command == "check":
        return selftest()
    try:
        if args.command == "report" and args.input:
            if any((args.voice, args.music, args.bed, args.sfx, args.mix)):
                parser.error("--input cannot be combined with stem arguments")
            if args.input == "-":
                result = json.load(sys.stdin)
            else:
                with Path(args.input).open(encoding="utf-8-sig") as source:
                    result = json.load(source)
            if result.get("schema") != 1:
                raise ValueError("Unsupported measurement JSON schema")
        else:
            if args.voice is None or args.music is None:
                parser.error("--voice and --music are required")
            result = inspect_mix(args.voice, args.music, args.bed, args.sfx, args.mix)
        if args.command == "report":
            print(markdown_report(result))
        elif args.json:
            print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
        else:
            print_stems(result)
        return 2 if any(m["status"] == "FAIL" for m in result["metrics"]) else 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            diagnostic = error.stderr
            if isinstance(diagnostic, bytes):
                diagnostic = diagnostic.decode(errors="replace")
            message = (diagnostic or str(error))[-2500:]
        else:
            message = str(error)
        print("ERROR: " + message, file=sys.stderr)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())

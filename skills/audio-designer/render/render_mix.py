"""CPU-only cue-sheet renderer: ffmpeg + numpy + stdlib; no network or models."""
import sys
sys.dont_write_bytecode = True  # Shared imports must never write into mix/ or scripts/.

import argparse
from contextlib import contextmanager
import csv
import hashlib
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import subprocess
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mix"))
import mix_check
audio_map = mix_check.audio_map

# Starting points, not standards. Attack/release are choices WITHIN the ranges
# on mix-plan.md:27; the master targets are from mix-plan.md:43.
# Ducking copies the cutter of two narrated shorts a listener approved:
# music threshold 0.03, attack 20, release 400; ambience threshold 0.01, attack 10, release 350, both against a
# voice at -26.7 LUFS. Thresholds are kept relative to the voice's integrated LUFS so a hotter voice ducks the same.
DUCK = {
    "music": {"threshold_db_re_voice": -3.8, "attack_ms": 20.0, "release_ms": 400.0},
    "bed": {"threshold_db_re_voice": -13.3, "attack_ms": 10.0, "release_ms": 350.0},
}
MIX_START = {
    "source": "../references/mix-plan.md:27,43",
    "integrated_lufs": -16.0,
    "true_peak_dbtp": -1.5,
    "lra_lu": 11.0,
    "lra_policy": "local loudnorm parameter, as in the existing video graphs; not a plan target",
}
SAMPLE_RATE = 48000
LAYERS = ("voice", "music", "bed", "sfx")
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ALIASES = {
    "gain": "gain_db", "gain_d_b": "gain_db", "duck": "duck_ratio",
    "cue_in_out": "cue", "cue_time": "cue", "excerpt_start_end": "excerpt",
    "fade_in_out": "fade", "duck_eq": "duck_eq",
    "eq_highpass_hz_lowpass_hz": "eq",
    "highpass": "highpass_hz", "lowpass": "lowpass_hz",
    "fade_in_s": "fade_in", "fade_out_s": "fade_out",
}
COLUMNS = {
    "layer", "file", "excerpt_start", "excerpt_end", "excerpt", "cue_in", "cue_out",
    "cue", "gain_db", "duck_ratio", "highpass_hz", "lowpass_hz", "fade_in", "fade_out",
    "fade", "duck_eq", "eq", "measured_after_gain", "listening",
}


@dataclass(frozen=True)
class Cue:
    row: int
    layer: str
    file: Path
    channels: int
    start_frame: int
    end_frame: int
    in_frame: int
    out_frame: int
    gain_db: float
    duck_ratio: float | None
    highpass_hz: float | None
    lowpass_hz: float | None
    fade_in: int
    fade_out: int

    @property
    def play_frames(self):
        return min(self.end_frame - self.start_frame, self.out_frame - self.in_frame)


class CommandLog:
    def __init__(self):
        self.commands = []
        self.stage = "probe"

    def run(self, args, **kwargs):
        entry = {"stage": self.stage, "argv": [str(a) for a in args]}
        self.commands.append(entry)
        try:
            result = subprocess.run(args, **kwargs)
            entry["returncode"] = result.returncode
            return result
        except subprocess.CalledProcessError as error:
            entry["returncode"] = error.returncode
            raise

    def ffmpeg(self, args):
        return self.run(args, check=True, capture_output=True,
                        encoding="utf-8", errors="replace")


@contextmanager
def record_shared_commands(log):
    """Also log the real probe/decoder/meter commands run by existing helpers.

    Replace only audio_map's module reference, not global subprocess.run.
    Like mix_check's decoder adapter, this is a single-threaded CLI context.
    """
    previous = audio_map.subprocess
    audio_map.subprocess = types.SimpleNamespace(run=log.run)
    try:
        yield
    finally:
        audio_map.subprocess = previous


def numeric(value, default=None):
    value = str(value).strip()
    if value in {"", "-"}:
        return default
    clean = re.sub(r"\s*(?:dB|Hz|seconds?|sec|s)\s*$", "", value, flags=re.I)
    result = float(clean)
    if not math.isfinite(result):
        raise ValueError("numbers must be finite")
    return result


def pair(value, separator=r"[-–]"):
    value = re.sub(r"\s*(?:seconds?|sec|s)\s*$", "", value.strip(), flags=re.I)
    found = re.fullmatch(rf"\s*({NUMBER})\s*(?:{separator}\s*({NUMBER}))?\s*", value)
    if not found:
        raise ValueError("expected seconds as START-END, or a single cue-in time")
    return float(found[1]), float(found[2]) if found[2] is not None else None


def ratio(value):
    text = str(value).strip()
    if text.endswith(":1"):
        text = text[:-2]
    result = numeric(text)
    if result is not None and not 1 <= result <= 20:
        raise ValueError("duck_ratio must be '-' or a ratio from 1:1 through 20:1")
    return result


def source_info(path):
    meta = audio_map.probe(path)  # Existing helper explicitly decodes ffprobe as UTF-8.
    audio = next((s for s in meta["streams"] if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise ValueError("source has no audio stream: " + str(path))
    channels = int(audio["channels"])
    if channels not in (1, 2):
        raise ValueError("only mono/stereo sources are supported: " + str(path))
    duration = None
    if audio.get("duration_ts") is not None and audio.get("time_base"):
        duration = float(int(audio["duration_ts"]) * Fraction(audio["time_base"]))
    if duration is None:
        duration = float(audio.get("duration") or meta["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("source needs a finite positive duration: " + str(path))
    return channels, round(duration * SAMPLE_RATE)


def read_cues(path):
    path = Path(path).expanduser().resolve(strict=True)
    cache, cues = {}, []
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("CSV needs a header and at least one cue")
        names = [re.sub(r"[^a-z0-9]+", "_", x.strip().lower()).strip("_")
                 for x in reader.fieldnames]
        names = [ALIASES.get(x, x) for x in names]
        if len(names) != len(set(names)) or set(names) - COLUMNS:
            raise ValueError("duplicate or unknown CSV columns: " + ", ".join(names))
        if not {"layer", "file"} <= set(names):
            raise ValueError("CSV needs layer and file columns")
        reader.fieldnames = names
        for raw in reader:
            line = reader.line_num
            if None in raw:
                raise ValueError(f"CSV row {line}: extra cells; quote commas in paths/EQ")
            if not any(str(v or "").strip() for v in raw.values()):
                continue
            row = {k: (v or "").strip() for k, v in raw.items()}
            try:
                layer = row["layer"].lower()
                if layer not in LAYERS:
                    raise ValueError("layer must be voice, music, bed or sfx")
                if not row["file"]:
                    raise ValueError("file is required")
                file = Path(row["file"]).expanduser()
                file = (file if file.is_absolute() else path.parent / file).resolve(strict=True)
                if not file.is_file():
                    raise ValueError("source is not a local file")
                if file not in cache:
                    cache[file] = source_info(file)
                channels, source_frames = cache[file]
                start, end = numeric(row.get("excerpt_start", ""), 0), numeric(row.get("excerpt_end", ""))
                if row.get("excerpt", "").lower() not in {"", "-", "whole"}:
                    start, end = pair(row["excerpt"])
                    if end is None:
                        raise ValueError("excerpt must be START-END or whole")
                end = source_frames / SAMPLE_RATE if end is None else end
                cue_in, cue_out = numeric(row.get("cue_in", ""), 0), numeric(row.get("cue_out", ""))
                if row.get("cue"):
                    cue_in, cue_out = pair(row["cue"])
                cue_out = cue_in + end - start if cue_out is None else cue_out
                gain_db = numeric(row.get("gain_db", ""), 0)
                duck_ratio = ratio(row.get("duck_ratio", "-"))
                hp, lp = numeric(row.get("highpass_hz", "")), numeric(row.get("lowpass_hz", ""))
                combined = row.get("duck_eq", "")
                if combined not in {"", "-", "in pause"}:
                    remainder = combined
                    for pattern, key in ((rf"duck\s*({NUMBER}(?::1)?)", "duck"),
                                         (rf"HP\s*({NUMBER})\s*(?:Hz)?", "hp"),
                                         (rf"LP\s*({NUMBER})\s*(?:Hz)?", "lp")):
                        found = re.search(pattern, remainder, re.I)
                        if found:
                            if key == "duck":
                                duck_ratio = ratio(found[1])
                            elif key == "hp":
                                hp = numeric(found[1])
                            else:
                                lp = numeric(found[1])
                            remainder = remainder[:found.start()] + remainder[found.end():]
                    if remainder.strip(" ,;/"):
                        raise ValueError("Duck / EQ supports duck N:1, HP N Hz and LP N Hz")
                if row.get("eq", "") not in {"", "-"}:
                    values = row["eq"].split("/")
                    if len(values) != 2:
                        raise ValueError("EQ must be highpass/lowpass Hz; use '-' to disable either")
                    hp, lp = numeric(values[0]), numeric(values[1])
                fade_in = numeric(row.get("fade_in", ""), .01)
                fade_out = numeric(row.get("fade_out", ""), .02)
                if row.get("fade", "") not in {"", "-"}:
                    fade_in, fade_out = pair(row["fade"], separator=r"[/–-]")
                    if fade_out is None:
                        raise ValueError("fade in/out needs two seconds values")
                if min(start, cue_in, fade_in, fade_out) < 0 or end <= start or cue_out <= cue_in:
                    raise ValueError("times must be nonnegative; excerpt/cue ends must follow starts")
                frames = [round(x * SAMPLE_RATE) for x in (start, end, cue_in, cue_out, fade_in, fade_out)]
                a, b, ci, co, fi, fo = frames
                if b > source_frames + 1 or b <= a or co <= ci:
                    raise ValueError("excerpt exceeds source or rounds to an empty cue")
                if fi + fo > min(b - a, co - ci):
                    raise ValueError("fades together exceed the audible cue length")
                if layer in {"voice", "sfx"} and duck_ratio is not None:
                    raise ValueError("duck_ratio applies only to music and bed")
                if any(v is not None and not 0 < v < SAMPLE_RATE / 2 for v in (hp, lp)):
                    raise ValueError("EQ cutoffs must be between 0 and 24000 Hz")
                if hp is not None and lp is not None and hp >= lp:
                    raise ValueError("highpass must be below lowpass")
                cues.append(Cue(line, layer, file, channels, a, b, ci, co, gain_db,
                                duck_ratio, hp, lp, fi, fo))
            except (ValueError, OSError, KeyError, StopIteration) as error:
                raise ValueError(f"CSV row {line}: {error}") from error
    if not cues:
        raise ValueError("CSV contains no cues")
    return path, cues


def align_filter(frames):
    return f"apad=whole_len={frames},atrim=end_sample={frames}"


def ffmpeg_base():
    return [audio_map.media_tool(), "-hide_banner", "-nostdin", "-y",
            "-filter_complex_threads", "1", "-filter_threads", "1"]


def inputs(paths):
    args = []
    for path in paths:
        args += ["-threads", "1", "-i", str(path)]
    return args


def wav_output(path):
    return ["-map", "[out]", "-vn", "-c:a", "pcm_f32le", "-ar", str(SAMPLE_RATE),
            "-ac", "2", "-threads", "1", str(path)]


def duck_graph(source, key, out, duck_ratio, layer, voice_lufs):
    spec = DUCK[layer]
    threshold = min(1.0, 10 ** ((voice_lufs + spec["threshold_db_re_voice"]) / 20))
    return (f"[{source}][{key}]sidechaincompress=threshold={threshold:.6g}:"
            f"ratio={duck_ratio}:attack={spec['attack_ms']}:"
            f"release={spec['release_ms']}[{out}]")


def render_layer(layer, cues, frames, directory, log):
    chosen = [cue for cue in cues if cue.layer == layer]
    paths = [cue.file for cue in chosen]
    ducked = [i for i, cue in enumerate(chosen) if cue.duck_ratio is not None]
    graph = []
    if ducked:
        voice_lufs = audio_map.loudness(directory / "voice.wav")["integrated_lufs"]
        paths.append(directory / "voice.wav")
        keys = "".join(f"[key{i}]" for i in ducked)
        splitter = f"asplit={len(ducked)}" if len(ducked) > 1 else "anull"
        # Explicitly pad the key even though the written voice stem is aligned.
        graph.append(f"[{len(chosen)}:a:0]apad=whole_len={frames},"
                     f"atrim=end_sample={frames},{splitter}{keys}")
    layers = []
    for i, cue in enumerate(chosen):
        filters = [f"aresample={SAMPLE_RATE}",
                   f"atrim=start_sample={cue.start_frame}:end_sample={cue.end_frame}",
                   "asetpts=PTS-STARTPTS", f"atrim=end_sample={cue.play_frames}"]
        if cue.channels == 1:
            filters.append("pan=stereo|c0=c0|c1=c0")  # Preserve mono amplitude in each channel.
        filters += ["aformat=sample_fmts=fltp:channel_layouts=stereo",
                    f"volume={cue.gain_db:.12g}dB"]
        if cue.highpass_hz is not None:
            filters.append(f"highpass=f={cue.highpass_hz}")
        if cue.lowpass_hz is not None:
            filters.append(f"lowpass=f={cue.lowpass_hz}")
        if cue.fade_in:
            filters.append(f"afade=t=in:ss=0:ns={cue.fade_in}")
        if cue.fade_out:
            filters.append(f"afade=t=out:ss={cue.play_frames - cue.fade_out}:ns={cue.fade_out}")
        filters += [f"adelay={cue.in_frame}S:all=1", align_filter(frames)]
        label = f"cue{i}"
        graph.append(f"[{i}:a:0]" + ",".join(filters) + f"[{label}]")
        if cue.duck_ratio is not None:
            key, output = f"key{i}", f"duck{i}"
            graph.append(duck_graph(label, key, output, cue.duck_ratio, layer, voice_lufs))
            label = output
        layers.append(f"[{label}]")
    if layers:
        graph.append("".join(layers) + f"amix=inputs={len(layers)}:normalize=0:duration=longest,"
                     + align_filter(frames) + "[out]")
    else:
        graph.append(f"anullsrc=r={SAMPLE_RATE}:cl=stereo,atrim=end_sample={frames}[out]")
    log.stage = layer
    log.ffmpeg(ffmpeg_base() + inputs(paths) + ["-filter_complex", ";".join(graph)]
               + wav_output(directory / (layer + ".wav")))


def loudnorm_filter(measured=None):
    filt = (f"loudnorm=I={MIX_START['integrated_lufs']}:TP={MIX_START['true_peak_dbtp']}:"
            f"LRA={MIX_START['lra_lu']}")
    if measured is not None:
        for option, key in (("measured_I", "input_i"), ("measured_TP", "input_tp"),
                            ("measured_LRA", "input_lra"), ("measured_thresh", "input_thresh"),
                            ("offset", "target_offset")):
            value = measured[key]
            if not math.isfinite(float(value)):
                raise ValueError("loudnorm cannot normalise silence/non-finite measurements")
            filt += f":{option}={value}"
        filt += ":linear=true"
    return filt + ":print_format=json"


def loudnorm_values(stderr):
    for match in reversed(list(re.finditer(r"\{[^{}]*\}", stderr))):
        value = json.loads(match[0])
        if "input_i" in value and "output_i" in value:
            return value
    raise ValueError("FFmpeg did not return loudnorm measurement JSON")


def render_final(directory, frames, log):
    args = ffmpeg_base() + inputs([directory / (layer + ".wav") for layer in LAYERS])
    summed = ("".join(f"[{i}:a:0]" for i in range(4))
              + "amix=inputs=4:normalize=0:duration=longest," + align_filter(frames) + ",")
    log.stage = "loudnorm_pass1"
    first = log.ffmpeg(args + ["-filter_complex", summed + loudnorm_filter() + "[out]",
                              "-map", "[out]", "-f", "null", "-"])
    measured = loudnorm_values(first.stderr)
    log.stage = "loudnorm_pass2"
    second = log.ffmpeg(args + ["-filter_complex", summed + loudnorm_filter(measured)
                               + f",aresample={SAMPLE_RATE}," + align_filter(frames) + "[out]"]
                        + wav_output(directory / "final.wav"))
    return {"pass1": measured, "pass2": loudnorm_values(second.stderr)}


def render_mix(cue_path, out):
    log = CommandLog()
    with record_shared_commands(log):
        cue_path, cues = read_cues(cue_path)
        directory = Path(out).expanduser()
        if directory.is_symlink() or (hasattr(directory, "is_junction") and directory.is_junction()):
            raise ValueError("output must not be a symlink/junction")
        directory = directory.resolve()
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError("output must be a new or empty directory; existing files are never overwritten")
        frames = max(cue.out_frame for cue in cues)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema": 1, "cue_csv": str(cue_path), "sample_rate": SAMPLE_RATE,
            "channels": 2, "frames": frames, "duration_s": frames / SAMPLE_RATE,
            "starting_values": MIX_START, "ducking": DUCK, "commands": log.commands,
            "shared_dependencies": {
                name: {"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
                for name, file in (("mix_check", Path(mix_check.__file__)),
                                   ("audio_map", Path(audio_map.__file__)),
                                   ("mix_plan", Path(__file__).resolve().parents[1] / "references" / "mix-plan.md"))
            },
            "cues": [{**asdict(cue), "file": str(cue.file)} for cue in cues],
            "stems": {layer: str(directory / (layer + ".wav")) for layer in LAYERS},
            "final": str(directory / "final.wav"),
            "notes": ["Measured evidence only; listening pending.",
                      "Music/bed stems include ducking; final uses those exact stems.",
                      "No implicit looping, stretching, source normalisation or recognition."],
        }
        try:
            for layer in LAYERS:
                render_layer(layer, cues, frames, directory, log)
            manifest["loudnorm"] = render_final(directory, frames, log)
            log.stage = "mix_check"
            result = mix_check.inspect_mix(directory / "voice.wav", directory / "music.wav",
                                           directory / "bed.wav", [directory / "sfx.wav"],
                                           directory / "final.wav")
            manifest["mix_check"] = result
            manifest["exit_code"] = 2 if any(m["status"] == "FAIL" for m in result["metrics"]) else 0
            (directory / "cue_report.md").write_text(mix_check.markdown_report(result) + "\n",
                                                    encoding="utf-8")
        except Exception as error:
            manifest["error"] = str(error)
            manifest["exit_code"] = 1
            raise
        finally:
            (directory / "render.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                encoding="utf-8")
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    render = commands.add_parser("render", help="render a local CSV, then run mix_check")
    render.add_argument("cue", type=Path)
    render.add_argument("--out", type=Path, required=True)
    commands.add_parser("check", help="synthetic integration checks with real renderer mutations")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            import render_tests
            return render_tests.run(sys.modules[__name__])
        result = render_mix(args.cue, args.out)
        fails = sum(m["status"] == "FAIL" for m in result["mix_check"]["metrics"])
        print(f"Rendered {result['duration_s']:.3f} s; mix_check: {fails} FAIL; listening pending.")
        return result["exit_code"]
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        diagnostic = error.stderr if isinstance(error, subprocess.CalledProcessError) else str(error)
        if isinstance(diagnostic, bytes):
            diagnostic = diagnostic.decode("utf-8", errors="replace")
        print("ERROR: " + (diagnostic or str(error))[-3000:], file=sys.stderr)
        return 1


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())

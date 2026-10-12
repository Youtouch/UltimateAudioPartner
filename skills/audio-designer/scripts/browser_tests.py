"""Deterministic browser HTTP checks; temporary files and cached vectors, no models."""
import sys
sys.dont_write_bytecode = True

from contextlib import redirect_stderr
import csv
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import audio_search
import sound_browser


def main():
    count, failures = 0, []

    def check(name, condition):
        nonlocal count
        count += 1
        print(("PASS " if condition else "FAIL ") + name)
        if not condition:
            failures.append(name)

    with tempfile.TemporaryDirectory(prefix="browser-api-") as temporary:
        root = Path(temporary)
        paths = ["Pack/approved.wav", "Pack/near.wav", "Pack/counter-close.wav", "Pack/clean.wav",
                 "Pack/rejected.wav", "Pack/vocal.wav", "OtherPack/approved.wav"]
        small = [(1, 0, 0), (.999, 0, math.sqrt(1 - .999 ** 2)), (.9, math.sqrt(.19), 0),
                 (.88, 0, math.sqrt(1 - .88 ** 2)), (.8, .6, 0),
                 (.98, 0, math.sqrt(1 - .98 ** 2)), (.2, 0, math.sqrt(.96))]
        audio = np.zeros((len(paths), 512), dtype="<f4")
        audio[:, :3] = small
        audio.tofile(root / "AUDIO_TAGS_clap.f32")
        fields = ["relative_path", "clap_row", "top_labels", "timeline", "speech", "music", "silence", "seconds_tagged"]
        with (root / "AUDIO_TAGS.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fields)
            writer.writeheader()
            for row, path in enumerate(paths):
                file = root / path
                file.parent.mkdir(exist_ok=True)
                file.write_bytes(b"synthetic API fixture; audio streaming is not exercised")
                writer.writerow({"relative_path": path, "clap_row": row, "top_labels": "Choir 0.90" if "vocal" in path else "Click 0.90",
                                 "timeline": "", "speech": 0, "music": 0, "silence": 0, "seconds_tagged": 5 + row})
        # Seed the real Index name cache to prove these routes need no embedding/model calls.
        names = sorted({audio_search.clean_name(path, path.split("/")[0]) for path in paths})
        np.savez(root / "AUDIO_NAMES_clap.npz", names=np.array(names), vectors=np.tile(audio[0], (len(names), 1)))
        cached = (root / "AUDIO_NAMES_clap.npz").stat().st_mtime_ns
        lib = sound_browser.Library(root)
        server = sound_browser.Server(("127.0.0.1", 0), sound_browser.handler(lib))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:" + str(server.server_address[1])

        def request(path, body=None):
            data = None if body is None else json.dumps(body).encode("utf-8")
            req = Request(base + path, data=data, headers={"Content-Type": "application/json"} if body is not None else {})
            try:
                with urlopen(req, timeout=10) as response:
                    raw = response.read().decode("utf-8")
                    return response.status, json.loads(raw) if response.headers.get_content_type() == "application/json" else raw
            except HTTPError as error:
                return error.code, json.loads(error.read().decode("utf-8"))

        def verdict(path, value):
            return request("/api/verdict", {"path": path, "verdict": value, "note": ""})

        try:
            note = io.StringIO()
            with redirect_stderr(note):
                status, payload = request("/api/data")
                request("/api/data")
            check("without a ledger the API lists every tagged file with duration and pack",
                  status == 200 and payload["total"] == len(paths)
                  and all(row[1] and row[3] == 5 + paths.index(row[0]) for row in payload["rows"]))
            check("missing-ledger fallback reports one visible note per library", note.getvalue().count("NOTE AUDIO_LEDGER_files.csv missing") == 1)
            status, empty = request("/api/approved")
            check("approved route with no approvals returns an actionable 400", status == 400 and bool(empty.get("error")))
            status, per_file = request("/api/like?" + urlencode({"p": paths[0]}))
            if status != 200: print("DIAGNOSTIC", status, per_file)
            check("per-file example route ranks its near duplicate first", status == 200
                  and per_file["results"][0]["path"] == paths[1])
            # Full paths are exact files here. The CLI's substring selection would wrongly select OtherPack too.
            check("per-file examples do not silently select a path containing the approved path", status == 200
                  and paths[6] in [row["path"] for row in per_file["results"]])
            status, _ = verdict(paths[0], "like")
            check("verdict POST records an approval", status == 200
                  and json.loads((root / "AUDIO_VERDICTS.json").read_text(encoding="utf-8"))[paths[0]]["verdict"] == "like")
            status, approved = request("/api/approved")
            if status != 200: print("DIAGNOSTIC", status, approved)
            check("saved approval reaches the real example scorer and near duplicate ranks first", status == 200
                  and approved["approved"] == [paths[0]] and approved["results"][0]["path"] == paths[1])
            before = [row["path"] for row in approved["results"]]
            check("counter-close candidate starts ahead of clean candidate", before.index(paths[2]) < before.index(paths[3]))
            verdict(paths[4], "reject")
            status, rejected = request("/api/approved")
            after = [row["path"] for row in rejected["results"]]
            check("saved rejection is passed as a counter-example and excluded", status == 200
                  and rejected["rejected"] == [paths[4]] and paths[4] not in after)
            check("counter-example penalty reverses the affected candidates", after.index(paths[3]) < after.index(paths[2]))
            by_path = {row["path"]: row for row in rejected["results"]}
            check("real scorer applies the independently calculated rejection penalty",
                  abs(by_path[paths[2]]["score"] - .859) < .002 and abs(by_path[paths[3]]["score"] - .88) < .002)
            scores = rejected["scores"]
            check("ranked API maps scores to the original fingerprint rows", len(scores) == len(paths)
                  and scores[0] is None and scores[4] is None and abs(scores[1] - .999) < .002)
            status, avoided = request("/api/approved?" + urlencode({"avoid": "Choir"}))
            check("avoid filters analysed labels", status == 200 and paths[5] not in [r["path"] for r in avoided["results"]])
            status, avoided = request("/api/like?" + urlencode({"p": paths[0], "avoid": "vocal"}))
            check("per-file avoid also filters names", status == 200 and paths[5] not in [r["path"] for r in avoided["results"]])
            status, _ = request("/api/approved?" + urlencode({"avoid": "["}))
            check("invalid avoid regex returns 400", status == 400)
            status, _ = request("/api/like?" + urlencode({"p": "../outside.wav"}))
            check("per-file endpoint rejects paths outside the library", status == 404)
            status, similar = request("/api/similar?row=0")
            check("audio-only similar endpoint remains available", status == 200 and abs(similar["scores"][1] - .999) < .002)
            check("example routes used only pre-existing cached vectors", cached == (root / "AUDIO_NAMES_clap.npz").stat().st_mtime_ns
                  and "audio_tag" not in sys.modules)

            for platform, os_name, command in (("win32", "nt", ["explorer", "/select,", str(root / paths[0])]),
                                               ("darwin", "posix", ["open", "-R", str(root / paths[0])]),
                                               ("linux", "posix", ["xdg-open", str((root / paths[0]).parent)])):
                # Only process launch is mocked; real HTTP routing, file validation and command selection run.
                with patch.object(sound_browser, "os", SimpleNamespace(name=os_name)), \
                     patch.object(sound_browser, "sys", SimpleNamespace(platform=platform)), \
                     patch.object(sound_browser.subprocess, "Popen") as launch:
                    status, _ = request("/api/reveal", {"path": paths[0]})
                    check(platform + " reveal selects its native command", status == 200 and launch.call_args.args[0] == command)

            # Measured words, duplicates, variation sets, examples and export, written as describe/groups/vocab passes would.
            import shutil
            import soundfile as sf
            import audio_describe
            sr = 48000
            click = np.concatenate([np.zeros(sr // 10), 0.2 * np.sin(2 * np.pi * 900 * np.arange(sr // 5) / sr), np.zeros(sr // 10)])
            sf.write(root / paths[3], click.astype("float32"), sr)
            desc = {p: dict.fromkeys(audio_describe.FIELDS, "") | {"relative_path": p, "describer": audio_describe.VERSION}
                    for p in paths}
            desc[paths[0]].update(sound_s=0.12, attack_ms=2, crest_db=18, centroid_hz=6000, onsets=1, decay_ms=90, lufs_momentary_max=-20)
            desc[paths[1]].update(sound_s=4.0, attack_ms=80, crest_db=6, centroid_hz=300, onsets=1, decay_ms=2000, lufs_momentary_max=-35)
            for k, p in enumerate(paths[2:], 2):
                desc[p].update(sound_s=1.0, centroid_hz=1000 + k, onsets=2)
            with (root / "AUDIO_DESCRIPTORS.csv").open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, audio_describe.FIELDS)
                writer.writeheader()
                writer.writerows(desc.values())
            with (root / "AUDIO_GROUPS.csv").open("w", newline="", encoding="utf-8") as target:
                target.write("relative_path,duplicate_of,duplicate_set,variation_set,variation_size\n"
                             f"{paths[0]},,d1,v1,3\n{paths[6]},{paths[0]},d1,,\n{paths[1]},,,v1,3\n{paths[2]},,,v1,3\n")
            with (root / "AUDIO_VOCAB_TAGS.csv").open("w", newline="", encoding="utf-8") as target:
                target.write("relative_path,tag,name,label,audio,sources,tagger,built_at\n")
                for p, tag in [(paths[0], "click"), (paths[1], "click"), (paths[2], "click"), (paths[3], "whoosh"), (paths[4], "whoosh"),
                               (paths[5], "coin"), (paths[5], "single_event"), (paths[4], "single_event"), (paths[3], "single_event")]:
                    target.write(f"{p},{tag},1,0.5,0.3,3,ced-base,now\n")
            status, data = request("/api/data")
            words = [w["word"] for w in data["words"]]
            row = {r[0]: r for r in data["rows"]}
            check("data carries the plain-word table with a sentence and a rule for every word", status == 200
                  and words[:3] == ["short", "long", "punchy"] and all(w["explain"] and w["rule"] for w in data["words"]))
            check("a short punchy bright sound earns those words from its measures, with the numbers behind them",
                  {"short", "punchy", "bright"} <= {words[k] for k in row[paths[0]][10]}
                  and row[paths[0]][11][data["shown"].index("attack_ms")] == 2)
            check("a long soft dark sound earns its own words and not the opposite ones",
                  {"long", "soft", "dark", "tail", "quiet"} <= {words[k] for k in row[paths[1]][10]}
                  and not {"short", "punchy", "bright"} & {words[k] for k in row[paths[1]][10]})
            check("every shown number has a plain sentence for Explain mode",
                  all(v[3] for m, v in data["numbers"].items() if m != "centroid_hz"))
            check("duplicates and variation sets reach the page", row[paths[6]][12][1] == paths[0] and row[paths[0]][12][2] == "v1")
            check("three example searches from the commonest agreed sound categories, character tags left out",
                  len(data["examples"]) == 3 and data["examples"][0].endswith("click") and data["examples"][1].endswith("whoosh")
                  and not any("single" in e for e in data["examples"]))
            with patch("sfx_groups.varied", lambda root_, path, n: [paths[2], paths[0]][:n]):
                status, varied = request("/api/varied?" + urlencode({"p": paths[0], "n": 2}))
            check("pick varied returns the chosen takes, ranked", status == 200 and varied["picked"] == [paths[2], paths[0]]
                  and varied["scores"][2] > varied["scores"][0])
            out = Path(temporary + "-export")
            try:
                status, exported = request("/api/export", {"paths": [paths[3]], "out": str(out), "format": "wav24", "rate": 44100,
                                                           "category": "ui"})
                check("export writes the selection outside the library, with a manifest", status == 200
                      and (out / exported["files"][0]).exists() and (out / "MANIFEST.csv").exists()
                      and sf.info(out / exported["files"][0]).subtype == "PCM_24")
                status, refused = request("/api/export", {"paths": [paths[3]], "out": str(root / "inside")})
                check("export into the library is refused", status == 400 and "outside" in refused["error"])
                status, _ = request("/api/export", {"paths": [], "out": str(out)})
                check("export with nothing ticked asks for a selection", status == 400)
            finally:
                shutil.rmtree(out, ignore_errors=True)

            class Hooks(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.ids = set()
                    self.avoid = self.explain = self.export = None
                    self.in_need, self.needs, self.explain_only = False, [], set()
                def handle_starttag(self, tag, attrs):
                    self.ids.update(value for key, value in attrs if key == "id")
                    if tag == "input" and dict(attrs).get("id") == "avoid":
                        self.avoid = dict(attrs)
                    if tag == "input" and dict(attrs).get("id") == "explain":
                        self.explain = dict(attrs)
                    if tag == "option" and self.in_need:
                        self.needs.append(dict(attrs).get("value"))
                    if tag == "select":
                        self.in_need = dict(attrs).get("id") == "need"
                    if "explain-only" in (dict(attrs).get("class") or "").split():
                        self.explain_only.add(dict(attrs).get("id") or tag)
                    if tag == "button" and dict(attrs).get("id") == "exportbtn":
                        self.export = dict(attrs)
            status, html = request("/")
            hooks = Hooks()
            hooks.feed(html)
            check("page exposes approved view and initially empty editable avoid field", status == 200
                  and {"approved", "avoid"}.issubset(hooks.ids)
                  and hooks.avoid is not None and hooks.avoid.get("value", "") == ""
                  and hooks.avoid.get("type") == "search" and 'data-act="morelike"' in html)
            check("Explain is one toggle, off by default", hooks.explain is not None and hooks.explain.get("type") == "checkbox"
                  and "checked" not in hooks.explain)
            check("'I need a sound for' offers six game events, each filling a search with plain words",
                  [v for v in hooks.needs if v] == ["short punchy click", "short bright pickup", "punchy impact", "short footsteps",
                                                     "deep explosion", "long loopable ambience"]
                  and all(set(v.split()) & set(words) for v in hooks.needs if v))
            check("beginner help stays hidden unless Explain is on", {"explainbox"} <= hooks.explain_only
                  and "body.explain #examples { display: flex; }" in html and "#examples { display: none;" in html)
            check("export, duplicates, word filters and examples have their hooks", hooks.export is not None and "disabled" in hooks.export
                  and {"exportdlg", "exportout", "choose", "showdups", "wordfacets", "exampleList", "need"} <= hooks.ids
                  and 'data-act="select"' in html and 'data-act="varied"' in html)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)

    print(f"{count - len(failures)}/{count} browser API tests passed; {len(failures)} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

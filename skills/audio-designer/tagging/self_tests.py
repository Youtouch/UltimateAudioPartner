"""Model-free regression tests; all fixture writes stay inside tagging."""
import csv
import io
import os
import tempfile
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_env] = "2"

import numpy as np

HERE = Path(__file__).resolve().parent
FIELDS = ["relative_path", "size_bytes", "mtime_ns", "seconds_tagged", "speech", "music",
          "silence", "top_labels", "timeline", "clap_row", "tagger"]


def entry(tag, synonyms, labels="", relation="direct"):
    return dict(tag=tag, name_synonyms=synonyms, prompts=["the sound of " + tag],
                audioset_labels=labels, mapping_relation=relation)


def csv_text(fields, rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def tag_row(path, row, **extra):
    result = dict.fromkeys(FIELDS, "")
    result.update(relative_path=path, clap_row=str(row), silence="0",
                  top_labels="Clicking 0.8", timeline="[]", tagger="ast")
    result.update(extra)
    return result


def names_labels(api):
    vocab = api.load_vocabulary()
    assert vocab and len({v["tag"] for v in vocab}) == len(vocab)
    for v in vocab:
        assert all(k in v for k in ("tag", "name_synonyms", "prompts", "audioset_labels", "mapping_relation"))
        assert isinstance(v["prompts"], list) and v["prompts"]
    click = entry("click", "click;clk;clack;tick", "Clicking;Tick;Clickety-clack")
    bird = entry("bird_song", "bird;chirp;chirping;tweet", "Bird;Bird vocalization, bird call, bird song")
    rain = entry("rain", "rain;raindrop", "Rain;Raindrop")
    insects = entry("insects", "insect;cricket", "Insect;Cricket")
    weak = entry("weak", "texture", "Texture", "approximate")
    for path in ("Pack/UI/Click_01.wav", "Pack/UI/CLK_02.wav", "Pack/UI/Clacks_03.wav",
                 "Pack/UI/Tick_04.wav", "click.wav"):
        assert api.match_name(path, "Pack" if "/" in path else "", click), path
    for path in ("Pack/Nature/Birds_01.wav", "Pack/Nature/Chirping_02.wav"):
        assert api.match_name(path, "Pack", bird), path
    assert api.match_name("Pack/Nature/Raindrops.wav", "Pack", rain)
    for path in ("Pack/Transport/Train.wav", "Pack/Transport/Training.wav"):
        assert not api.match_name(path, "Pack", rain), path
    assert not api.match_name("Pack/no_birds.wav", "Pack", bird)
    assert not api.match_name("ClickPack/Train.wav", "ClickPack", click)
    assert not api.match_name("Pack/Clickable.wav", "Pack", click)
    mapped, approximate = api.map_labels(
        "Clicking 0.31; Tick 0.52; Bird vocalization, bird call, bird song 0.72; "
        "Bird 0.30; Insect 0.8; Cricket 0.9; Texture 0.93", [click, bird, rain, insects, weak])
    assert mapped.shape == approximate.shape == (5,)
    assert np.allclose(mapped[[0, 1, 3]], [0.52, 0.72, 0.9]), "parent/child labels collapse by maximum"
    assert np.isnan(mapped[[2, 4]]).all()
    assert np.isnan(approximate[:4]).all() and np.isclose(approximate[4], 0.93)
    comma, _ = api.map_labels("Clicking 0.20, Tick 0.40", [click])
    assert np.isclose(comma[0], 0.4)
    absent, absent_weak = api.map_labels("", [click, weak])
    assert np.isnan(absent).all() and np.isnan(absent_weak).all()


def library(api):
    with tempfile.TemporaryDirectory(prefix="_selftest_", dir=HERE) as directory:
        root = Path(directory)
        rows = [
            tag_row("Pack/Click.wav", 0), tag_row("Pack/Clack.wav", 2),
            tag_row("Pack/Bird.wav", 2), tag_row("Pack/Rain.wav", 3),
            tag_row("Pack/Missing.wav", ""), tag_row("Pack/Zero.wav", 4),
            tag_row("Pack/Partial.wav", 5), tag_row("TagsOnly/Tick.wav", 0), tag_row("click.wav", 0),
            tag_row("Pack/Click.wav", 1, tagger="ced-base", top_labels="Clicking 0.9",
                    timeline='[{"note":"complete\nquoted field"}]'),
            tag_row("Pack/Clack.wav", 999, tagger="ced-base"),
        ]
        paths = ["Pack/Click.wav", "Pack/Clack.wav", "Pack/Bird.wav", "Pack/Rain.wav",
                 "Pack/Missing.wav", "Pack/Zero.wav", "Pack/Partial.wav", "Pack/LedgerOnly.wav"]
        (root / "AUDIO_LEDGER_files.csv").write_text("\ufeff" + csv_text(
            ["relative_path", "pack"], [dict(relative_path=p, pack="Pack") for p in paths]), encoding="utf-8")
        (root / "AUDIO_TAGS.csv").write_text(csv_text(FIELDS, rows) +
            'Pack/Unfinished.wav,1,2,3,0,0,0,"unterminated', encoding="utf-8")
        vectors = np.zeros((5, 512), dtype="<f4")
        vectors[:4, :4] = np.eye(4, dtype="<f4")
        raw = vectors.tobytes() + np.ones(100, dtype="<f4").tobytes() + b"\x01\x02"
        (root / "AUDIO_TAGS_clap.f32").write_bytes(raw)
        lib = api.load_library(root)
        assert set(lib.paths) == set(paths) | {"TagsOnly/Tick.wav", "click.wav"}, "all-file coverage"
        assert len(lib.paths) == len(set(lib.paths))
        assert isinstance(lib.rows, list) and all(isinstance(r, dict) for r in lib.rows)
        assert isinstance(lib.meta, dict)
        assert lib.audio.shape == (len(lib.paths), 512)
        assert lib.valid_audio.shape == lib.silence.shape == (len(lib.paths),)
        assert lib.valid_audio.dtype == np.bool_
        assert len(lib.rows) == len(lib.packs) == len(lib.paths)
        idx = dict(zip(lib.paths, range(len(lib.paths))))
        root_click = idx["click.wav"]
        assert api.match_name(lib.paths[root_click], lib.packs[root_click], entry("click", "click")), \
            "root-level filenames must survive loaded pack normalization"
        first = idx["Pack/Click.wav"]
        assert lib.rows[first]["tagger"] == "ced-base" and lib.valid_audio[first]
        assert lib.rows[first]["top_labels"] == "Clicking 0.9"
        assert np.array_equal(lib.audio[first], vectors[1])
        assert lib.rows[idx["Pack/Clack.wav"]]["tagger"] == "ced-base"
        for path in ("Pack/Clack.wav", "Pack/Missing.wav", "Pack/Zero.wav", "Pack/Partial.wav", "Pack/LedgerOnly.wav"):
            assert not lib.valid_audio[idx[path]], "invalid latest row must not fall back: " + path
        assert lib.valid_audio[idx["TagsOnly/Tick.wav"]]
        assert (root / "AUDIO_TAGS_clap.f32").read_bytes() == raw, "truncate in memory only"
        temp = root / "replacement.csv"
        temp.write_text(csv_text(FIELDS, rows + [tag_row("Pack/Click.wav", 3,
            tagger="ced-after-swap", top_labels="Rain 0.99")]), encoding="utf-8")
        os.replace(temp, root / "AUDIO_TAGS.csv")
        fresh = api.load_library(root)
        first = fresh.paths.index("Pack/Click.wav")
        assert fresh.rows[first]["tagger"] == "ced-after-swap"
        assert np.array_equal(fresh.audio[first], vectors[3])
        (root / "AUDIO_LEDGER_files.csv").unlink()
        assert "TagsOnly/Tick.wav" in api.load_library(root).paths
        (root / "AUDIO_TAGS.csv").unlink()
        (root / "AUDIO_LEDGER_files.csv").write_text(csv_text(["relative_path", "pack"],
            [dict(relative_path="Only/Unprocessed.wav", pack="Only")]), encoding="utf-8")
        only = api.load_library(root)
        assert only.paths == ["Only/Unprocessed.wav"] and not only.valid_audio.any()


def audio_rules(api):
    base = np.r_[[0.95, 0.93, 0.91, 0.89, 0.87], np.linspace(0.15, 0.05, 995)]
    valid, silence = np.ones(1000, dtype=bool), np.zeros(1000)
    scores = np.column_stack([base, base * 0.03 + 0.22, np.full(1000, 0.99)])
    supports, rules = api.tag_audio_decisions(scores, valid, silence, ["click", "bird", "constant"], top_k=20)
    assert supports.shape == scores.shape and supports.dtype == np.bool_
    assert isinstance(rules, list) and len(rules) == 3 and all(isinstance(r, dict) for r in rules)
    assert supports[:5, 0].all() and 5 <= supports[:, 0].sum() < 20
    assert np.array_equal(supports[:, 0], supports[:, 1]), "per-tag affine invariance"
    assert not supports[:, 2].any(), "constant scores abstain"
    assert (supports.sum(axis=0) <= 20).all()
    linear = np.column_stack([np.linspace(1, 0, 1000), np.linspace(0.4, 0.2, 1000)])
    assert not api.tag_audio_decisions(linear, valid, silence, ["a", "b"], top_k=2)[0].any()
    weak = np.linspace(1, 0, 1000)
    weak[20:] -= 0.0001
    assert not api.tag_audio_decisions(weak[:, None], valid, silence, ["weak"], top_k=1)[0].any()
    tied = np.column_stack([base] * 4)
    capped, _ = api.tag_audio_decisions(tied, valid, silence, ["a", "b", "c", "d"], top_k=2)
    again, _ = api.tag_audio_decisions(tied, valid, silence, ["a", "b", "c", "d"], top_k=2)
    assert capped.any() and (capped.sum(axis=0) <= 2).all() and np.array_equal(capped, again)
    one, _ = api.tag_audio_decisions(tied, valid, silence, ["a", "b", "c", "d"], top_k=1)
    assert (one.sum(axis=0) <= 1).all()
    valid[0] = False
    silence[[1, 2, 3]] = [0.5, 0.499, 0.95]
    guarded, _ = api.tag_audio_decisions(np.column_stack([base, base]), valid, silence,
        ["click", "silence_or_room_tone"], top_k=20)
    assert not guarded[0].any()
    assert not guarded[[1, 3], 0].any() and guarded[2, 0]
    assert guarded[[1, 3], 1].all()


def statistics(api):
    support = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1],
                        [1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool)[:, None, :]
    support = np.concatenate([support, np.zeros_like(support)], axis=1)
    active, empty = api.source_statistics(support, ["active", "empty"])
    assert (active["files_1"], active["files_2"], active["files_3"]) == (3, 3, 1)
    assert (active["name_files"], active["label_files"], active["audio_files"]) == (5, 3, 4)
    expected = dict(name_to_label=2/5, label_to_name=2/3, name_to_audio=2/5, audio_to_name=2/4,
                    label_to_audio=2/3, audio_to_label=2/4,
                    name_loo_precision=3/5, label_loo_precision=1, audio_loo_precision=3/4)
    for key, value in expected.items():
        assert np.isclose(active[key], value), (key, active[key], value)
        assert empty[key] is None, "zero denominators must be explicit nulls: " + key
    assert (empty["files_1"], empty["files_2"], empty["files_3"]) == (0, 0, 0)


def review(api):
    vocabulary = [entry("click", "click"), entry("bird_song", "bird"), entry("rain", "rain")]

    def fixture(paths, packs=None, rows=None, valid=None):
        n = len(paths)
        return api.Library(
            paths=paths, packs=packs if packs is not None else [p.split("/")[0] for p in paths],
            rows=rows if rows is not None else [dict(tagger="fixture", top_labels="Clicking 0.8")] * n,
            audio=np.ones((n, 512), dtype=np.float32),
            valid_audio=np.ones(n, dtype=bool) if valid is None else np.array(valid, dtype=bool),
            silence=np.zeros(n), meta={})

    def pick(lib, support, top=150):
        shape = support.shape[:2]
        return api.review_candidates(lib, vocabulary[:shape[1]], support, np.zeros(shape),
            np.full(shape, np.nan), [dict(tag=v["tag"]) for v in vocabulary[:shape[1]]], top=top)

    # A file supporting two different singleton tags still receives just one review row.
    shared = fixture(["Shared/ClickBird.wav"])
    two_tags = np.zeros((1, 2, 3), dtype=bool)
    two_tags[0, :, 0] = True
    assert len(pick(shared, two_tags)) == 1, "review must contain one row per file"
    # Two or three supporting sources are not singletons and must never enter the queue.
    for source_count in (2, 3):
        corroborated = np.zeros((1, 1, 3), dtype=bool)
        corroborated[0, 0, :source_count] = True
        assert pick(shared, corroborated) == [], "corroborated evidence is not a singleton"

    variants = fixture(["Bundle/Click_01.wav", "Bundle/Click_02.wav", "Bundle/Rain_01.wav"])
    variant_support = np.zeros((3, 1, 3), dtype=bool)
    variant_support[:, 0, 0] = True
    selected = pick(variants, variant_support)
    assert len(selected) == 2
    assert sum("/Click_" in row["relative_path"] for row in selected) == 1, "variant families must be unique"

    codecs = fixture(["Pack/WAV/Ambience/Signal.wav", "Pack/MP3/Ambience/Signal.mp3"], packs=["Pack", "Pack"])
    codec_support = np.zeros((2, 1, 3), dtype=bool)
    codec_support[:, 0, 0] = True
    assert len(pick(codecs, codec_support)) == 1, "WAV/MP3 copies are one review family"

    paths, packs = [], []
    for i in range(60):
        group = i // 8
        word = chr(ord("a") + i // 26) + chr(ord("a") + i % 26)
        paths.append("Folder" + str(group) + "/Click_" + word + ".wav")
        display = ("Bundle" + chr(ord("A") + group))
        packs.append(display if i % 3 == 0 else display.lower() + " (" + str(i % 3) + ")")
    quota_lib = fixture(paths, packs=packs)
    quota_support = np.zeros((60, 3, 3), dtype=bool)
    quota_support[np.arange(60), np.arange(60) % 3, 0] = True
    full = pick(quota_lib, quota_support)
    assert len(full) >= 10 and len({row["relative_path"] for row in full}) == len(full)
    for tag in ("click", "bird_song", "rain"):
        assert sum(row["tag"] == tag for row in full) <= 10, "review tag quota"
    for group in range(8):
        normalized = "bundle" + chr(ord("a") + group)
        assert sum(row["pack"].split(" (")[0].strip().lower() == normalized for row in full) <= 5, \
            "review quota must combine pack aliases and numbered copies"
    limited = pick(quota_lib, quota_support, top=3)
    assert len(limited) == 3 and [row["rank"] for row in limited] == [1, 2, 3], "--top must cap selection"

    reasons_lib = fixture(["Missing/Click.wav", "Printed/Click.wav"],
        rows=[{}, dict(tagger="fixture", top_labels="Rain 0.8")], valid=[False, True])
    reasons_support = np.zeros((2, 1, 3), dtype=bool)
    reasons_support[:, 0, 0] = True
    reasons = {row["relative_path"]: row["reason"].lower() for row in pick(reasons_lib, reasons_support)}
    assert "audio not evaluated" in reasons["Missing/Click.wav"], "missing audio must not be described as negative evidence"
    assert "labels not evaluated" in reasons["Missing/Click.wav"]
    assert "unknown" in reasons["Printed/Click.wav"], "top-label truncation does not establish absence"


def review_attribution(api):
    vocabulary = [entry("instrumental", "instrumental"),
                  entry("choir", "choir"), entry("calm", "calm")]
    lib = api.Library(
        paths=["Pack/Unspecified.wav"], packs=["Pack"],
        rows=[dict(tagger="fixture", top_labels="")],
        audio=np.ones((1, 512), dtype=np.float32), valid_audio=np.ones(1, dtype=bool),
        silence=np.zeros(1), meta={})
    rules = [dict(tag=v["tag"]) for v in vocabulary]

    def selected(support, labels=None):
        return api.review_candidates(lib, vocabulary, support, np.ones((1, 3)),
            np.full((1, 3), np.nan) if labels is None else labels, rules, top=10)

    # Instrumental audio support says nothing about whether the filename claimed instrumental.
    models_only = np.zeros((1, 3, 3), dtype=bool)
    models_only[0, 0, 2] = True
    models_only[0, 1, 1] = True
    labels = np.full((1, 3), np.nan)
    labels[0, 1] = 0.8
    rows = selected(models_only, labels)
    assert rows and all(row["priority"] < 4 for row in rows), "model-only evidence must not add a name tension"
    assert all("instrumental name versus" not in row["reason"].lower() and
               "instrumental filename" not in row["reason"].lower() for row in rows)

    # A choir filename is name evidence, not a model vocal-presence claim.
    names_only = np.zeros((1, 3, 3), dtype=bool)
    names_only[0, 0, 0] = True
    names_only[0, 1, 0] = True
    rows = selected(names_only)
    assert rows and all(row["priority"] < 4 for row in rows), "two filename claims must not add a model/name tension"
    assert all("model vocal" not in row["reason"].lower() for row in rows)

    # The calm/voice question also requires actual model voice evidence.
    names_only[0, 0, 0] = False
    names_only[0, 2, 0] = True
    rows = selected(names_only)
    assert rows and all(row["priority"] < 4 for row in rows), "name-only choir/calm must not add a model voice tension"
    assert all("vocal-layer question" not in row["reason"].lower() for row in rows)


def review_tagger_priority(api):
    vocabulary = [entry("click", "click", "Clicking")]
    lib = api.Library(
        paths=["Pack/ASTLow.wav", "Pack/ASTTop.wav", "Pack/CEDLow.wav", "Pack/CEDTop.wav"],
        packs=["Pack"] * 4,
        rows=[dict(tagger=tagger, top_labels="Clicking " + str(value))
              for tagger, value in zip(["ast", "ast", "ced-base", "ced-base"], [0.1, 0.2, 0.8, 0.9])],
        audio=np.ones((4, 512), dtype=np.float32), valid_audio=np.ones(4, dtype=bool),
        silence=np.zeros(4), meta={})
    support = np.zeros((4, 1, 3), dtype=bool)
    support[:, 0, 1] = True
    rows = api.review_candidates(lib, vocabulary, support, np.zeros((4, 1)),
        np.array([[0.1], [0.2], [0.8], [0.9]]), [dict(tag="click")], top=10)
    assert {row["relative_path"] for row in rows} == {"Pack/ASTTop.wav", "Pack/CEDTop.wav"}, \
        "label review priority must rank inside each tagger cohort rather than compare raw AST/CED scores"


def run():
    if not __debug__:
        raise RuntimeError("Run check without Python -O; assertions are required.")
    try:
        import vocab_tags as api
    except ImportError as exc:
        raise AssertionError("vocab_tags implementation is not yet available") from exc
    for test in (names_labels, library, audio_rules, statistics, review, review_attribution, review_tagger_priority):
        test(api)
        print(test.__name__ + ": ok")
    from label_self_tests import run as run_labels
    run_labels(api)
    print("All vocabulary tagging self-tests passed.")


if __name__ == "__main__":
    run()


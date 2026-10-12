"""Fit label decisions for tags added to the vocabulary, and measure tag agreement by sound length.

    python game_vocab.py fit --root <library>             # writes label_decisions_v3.json beside this script
    python game_vocab.py compare --root <library> --before OLD/AUDIO_VOCAB_TAGS.csv --after NEW/AUDIO_VOCAB_TAGS.csv
    python game_vocab.py check

`fit` keeps every rule of the frozen v2 policy unchanged and fits only the tags v2 does not know, with the v2
recipe, on DEV packs only (the same pack split as v2). Cohorts without rows in this library (an older tagger)
get the sparse-policy flat rule for the new tags. Nothing here decodes audio or runs the tagger: it reads
AUDIO_TAGS.csv, the cached fingerprints and the CLAP text prompts, CPU only.

`compare` reports, per split and length bucket, how often a second source confirms a name tag, and how many
files carry at least one agreed tag (two or more sources). Agreement is consistency between sources, not accuracy.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import vocab_tags as api  # sets offline, CPU-only and thread limits before numpy
import numpy as np
import label_rules as lr

HERE = Path(__file__).resolve().parent
V2, V3 = HERE / "label_decisions_v2.json", HERE / "label_decisions_v3.json"
BUCKETS = ((0, .3, "<0.3 s"), (.3, 1, "0.3-1 s"), (1, 10, "1-10 s"), (10, float("inf"), ">=10 s"))


def splits_for(packs):
    """v2's DEV/TEST assignment where it exists (so no pack changes side), the same seeded hash elsewhere."""
    fixed, path = {}, HERE / "LABEL_SPLITS_v2.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            fixed = {r["normalized_pack"]: r["split"] for r in csv.DictReader(f)}
    return lr.pack_splits(packs, fixed)


def extend_policy(old, vocabulary, fitted):
    """v2 rules for v2's tags, `fitted` (cohort -> tag -> rule) for the rest, flat fallback where none was fitted."""
    known = {cohort: {r["tag"]: r for r in rules} for cohort, rules in old["cohorts"].items()}
    flat = lambda tag: dict(tag=tag, threshold=.1, min_persistence=0., status="sparse_uncalibrated_flat_fallback",
                            dev_candidates=0, dev_confirmed=0, dev_confirmed_packs=0)
    cohorts = {}
    for cohort in sorted(set(known) | set(fitted)):
        cohorts[cohort] = [known.get(cohort, {}).get(v["tag"]) or fitted.get(cohort, {}).get(v["tag"]) or flat(v["tag"])
                           for v in vocabulary]
    return {**old, "vocabulary_signature": lr.vocabulary_signature(vocabulary), "cohorts": cohorts,
            "extended_from": {"policy": V2.name, "sha256": api.digest(V2.read_bytes()), "kept_tags": len(next(iter(known.values()))),
                              "new_tags": [v["tag"] for v in vocabulary if v["tag"] not in next(iter(known.values()))]},
            "extension_note": "v2 rules unchanged; new tags fitted with the v2 recipe on DEV packs only; TEST unused."}


def fit(root):
    if V3.exists():
        raise FileExistsError(f"{V3.name} exists: keep it, or delete it deliberately to refit")
    old = json.loads(V2.read_text(encoding="utf-8"))
    library, vocabulary = api.load_library(root), api.load_vocabulary()
    tags = [v["tag"] for v in vocabulary]
    v2_tags = {r["tag"] for r in next(iter(old["cohorts"].values()))}
    new = [j for j, t in enumerate(tags) if t not in v2_tags]
    patterns = [api.synonym_patterns(v) for v in vocabulary]
    name = np.array([[api.token_match(api.name_tokens(p, pack), pat) for pat in patterns]
                     for p, pack in zip(library.paths, library.packs)], dtype=bool)
    text, _ = api.prompt_vectors(vocabulary)
    scores = library.audio @ text.T
    scores[~library.valid_audio] = np.nan
    audio, _ = api.tag_audio_decisions(scores, library.valid_audio, library.silence, tags)
    reference = name | audio
    split = splits_for(library.packs)
    fitted, report = {}, {}
    cohorts = {}
    for i, row in enumerate(library.rows):
        if row:
            cohorts.setdefault(row.get("tagger") or "legacy-unspecified", []).append(i)
    for cohort, members in cohorts.items():
        members = np.array(members)
        sub = [library.rows[i] for i in members]
        label_scores, persistence, _ = lr.label_features(sub, [vocabulary[j] for j in new])
        rules = lr.fit_on_dev(label_scores, persistence, reference[members][:, new], [library.packs[i] for i in members],
                              [tags[j] for j in new], split[members], old["recipe"])
        fitted[cohort] = {r["tag"]: r for r in rules}
        report[cohort] = {r["tag"]: r["status"] for r in rules}
    bundle = extend_policy(old, vocabulary, fitted)
    V3.write_text(json.dumps(bundle, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=1))
    print(f"wrote {V3.name}: {len(new)} new tags fitted on DEV ({int((split == 'dev').sum())} files)")


def rates(rows, durations, split, paths_split, derived=False):
    """Per length bucket: name tags confirmed by a second source, and files with an agreed tag."""
    by_file = {}
    for r in rows:
        if not derived and r["tag"].startswith("footsteps_on_"):
            continue
        by_file.setdefault(r["relative_path"], []).append(r)
    out = {}
    for lo, hi, label in BUCKETS:
        files = [p for p, s in paths_split.items() if s == split and lo <= durations.get(p, -1) < hi]
        name_pairs = confirmed = label_conf = agreed_files = 0
        for p in files:
            tags = by_file.get(p, [])
            agreed_files += any(int(t["sources"]) >= 2 for t in tags)
            for t in tags:
                if t["name"] == "1":
                    name_pairs += 1
                    confirmed += int(t["sources"]) >= 2
                    label_conf += t["label"] != ""
        out[label] = {"files": len(files), "files_with_agreed_tag": agreed_files,
                      "agreed_file_rate": round(agreed_files / len(files), 4) if files else None,
                      "name_tags": name_pairs, "name_confirmed": round(confirmed / name_pairs, 4) if name_pairs else None,
                      "name_confirmed_by_label": round(label_conf / name_pairs, 4) if name_pairs else None}
    return out


def compare(root, before, after):
    library = api.load_library(root)
    ledger = {r["relative_path"]: r for r in api.read_csv_snapshot(Path(root) / "AUDIO_LEDGER_files.csv")[0]}
    durations = {p: api.numeric(ledger.get(p, {}).get("duration_seconds")) for p in library.paths}
    paths_split = dict(zip(library.paths, splits_for(library.packs)))
    read = lambda path: api.read_csv_snapshot(Path(path))[0]
    old, new = read(before), read(after)
    result = {s: {"before": rates(old, durations, s, paths_split), "after": rates(new, durations, s, paths_split),
                  "after_with_footstep_materials": rates(new, durations, s, paths_split, derived=True)}
              for s in ("dev", "test")}
    print(json.dumps(result, indent=1))
    return result


def check():
    old = {"recipe": "adaptive", "version": lr.VERSION,
           "cohorts": {"ast": [dict(tag="click", threshold=.4, min_persistence=0., status="dev")]}}
    vocabulary = [dict(tag="click", audioset_labels="Clicking", mapping_relation="direct", name_synonyms="click"),
                  dict(tag="laser", audioset_labels="Zing", mapping_relation="approximate", name_synonyms="laser")]
    global V2
    keep, V2 = V2, Path(__file__)  # extend_policy hashes its source policy; any readable file stands in here
    try:
        bundle = extend_policy(old, vocabulary, {"ced-base": {"laser": dict(tag="laser", threshold=.3, min_persistence=0.,
                                                                           status="dev_consistency_optimized")}})
    finally:
        V2 = keep
    assert [r["threshold"] for r in bundle["cohorts"]["ast"]] == [.4, .1], "old rule kept, unknown new tag falls back flat"
    assert [r["threshold"] for r in bundle["cohorts"]["ced-base"]] == [.1, .3], bundle["cohorts"]["ced-base"]
    assert bundle["vocabulary_signature"] == lr.vocabulary_signature(vocabulary)
    rows = [dict(relative_path="a.wav", tag="click", name="1", label="", sources="2"),
            dict(relative_path="a.wav", tag="footsteps_on_gravel", name="1", label="", sources="2"),
            dict(relative_path="b.wav", tag="laser", name="1", label="", sources="1"),
            dict(relative_path="c.wav", tag="laser", name="1", label=".5", sources="2")]
    r = rates(rows, {"a.wav": .2, "b.wav": .2, "c.wav": 5}, "test", {"a.wav": "test", "b.wav": "test", "c.wav": "dev"})
    assert r["<0.3 s"] == {"files": 2, "files_with_agreed_tag": 1, "agreed_file_rate": .5, "name_tags": 2,
                           "name_confirmed": .5, "name_confirmed_by_label": 0.0}, r["<0.3 s"]
    assert r["1-10 s"]["files"] == 0, "a DEV file never counts on TEST"
    print("PASS game_vocab check: v2 rules kept, new tags fitted or flat, agreement by length bucket")


def main():
    parser = argparse.ArgumentParser(
        description='Fit new vocabulary rules on DEV and compare agreement by sound length.',
        epilog='Example: python tagging/game_vocab.py compare --root MySounds --before old.csv --after new.csv',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True, title='commands', help='available commands')
    f = sub.add_parser(
        'fit',
        description='Fit only new-tag decisions on DEV packs; preserve existing frozen rules.',
        epilog='Example: python tagging/game_vocab.py fit --root MySounds',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='Fit only new-tag decisions on DEV packs; preserve existing frozen rules.',
    )
    f.add_argument("--root", type=Path, required=True, help='folder containing the audio library')
    c = sub.add_parser(
        'compare',
        description='Compare canonical-tag agreement by length bucket on DEV and TEST.',
        epilog='Example: python tagging/game_vocab.py compare --root MySounds --before old.csv --after new.csv --out comparison.json',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='Compare canonical-tag agreement by length bucket on DEV and TEST.',
    )
    c.add_argument("--root", type=Path, required=True, help='folder containing the audio library')
    c.add_argument("--before", type=Path, required=True, help='canonical-tag CSV from before the rule change')
    c.add_argument("--after", type=Path, required=True, help='canonical-tag CSV from after the rule change')
    c.add_argument("--out", type=Path, help='optional JSON comparison report path')
    sub.add_parser(
        'check',
        description='Run synthetic checks without a library or model download.',
        epilog='Example: python tagging/game_vocab.py check',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help='Run synthetic checks without a library or model download.',
    )
    a = parser.parse_args()
    if a.cmd == "check":
        return check()
    if a.cmd == "fit":
        return fit(a.root)
    result = compare(a.root, a.before, a.after)
    if a.out:
        a.out.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

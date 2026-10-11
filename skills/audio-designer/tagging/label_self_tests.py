"""New-rule regression fixtures, with independently hand-counted expectations."""
import numpy as np
from pathlib import Path


def label_thresholds(api):
    # A global threshold or a transposed tag axis must change these votes.
    assert (Path(__file__).parent / "label_rules.py").exists(), "per-tag label decisions are not implemented"
    import label_rules as lr
    scores = np.array([[.4, .4], [.8, .8], [.2, np.nan]], dtype=np.float32)
    rules = [{"tag": "a", "threshold": .3, "min_persistence": 0},
             {"tag": "b", "threshold": .7, "min_persistence": 0}]
    got = lr.apply_rules(scores, np.ones_like(scores), rules)
    assert got.tolist() == [[True, False], [True, True], [False, False]], "thresholds must follow each tag"
    rules[0]["min_persistence"] = .5
    persistence = np.array([[1/6, 1], [5/6, 1], [np.nan, 1]])
    got = lr.apply_rules(scores, persistence, rules)
    assert got.tolist() == [[False, False], [True, True], [False, False]]


def label_windows(api):
    # Two mapped children and a duplicated interval are still ONE evidence window.
    import label_rules as lr
    vocabulary = [dict(tag="bird", audioset_labels="Bird;Chirp, tweet", mapping_relation="direct"),
                  dict(tag="weak", audioset_labels="Bird", mapping_relation="approximate")]
    timeline = ("0-10: Bird 0.8, Chirp, tweet 0.7 | 0-10: Bird 0.9 | "
                "10-20: Music 0.9 | 20-30: Bird 0.3 | 30-40: | "
                "40-50: Music 0.7 | 50-60: Music 0.8")
    counts, totals = lr.window_counts(timeline, vocabulary)
    assert totals == 6 and counts.tolist() == [2, 0], "count windows, not labels or duplicate intervals"
    counts, totals = lr.window_counts("0-10: Music 0.9", vocabulary)
    assert totals == 1 and counts.tolist() == [0, 0]
    assert lr.window_counts("[]", vocabulary)[1] == 0, "unavailable timeline stays unknown"
    rules = [dict(threshold=.1, min_persistence=.5)]
    got = lr.apply_rules(np.array([[.8], [.8], [.8]]), np.array([[1/6], [5/6], [np.nan]]), rules)
    assert got.tolist() == [[False], [True], [True]], "unknown timeline must not invent an absence"


def label_split_isolation(api):
    # Per-file assignment, pack copy aliases, or silently mixed bench packs must fail.
    import label_rules as lr
    packs = ["Bundle (1)", "bundle", "Other", "Other", "Third"]
    split = lr.pack_splits(packs, {"bundle": "dev", "other": "test"})
    assert split[:4].tolist() == ["dev", "dev", "test", "test"]
    assert lr.pack_splits(list(reversed(packs)), {"bundle": "dev", "other": "test"}).tolist() == list(reversed(split.tolist()))
    try:
        lr.assert_pack_isolation(packs, np.array(["dev", "test", "test", "test", "dev"]))
    except AssertionError:
        pass
    else:
        assert False, "mixed pack splits must fail loudly"


def label_fit_and_cost(api):
    # A filter that drops all claims or a fitter using TEST scores must fail.
    import label_rules as lr
    scores = np.r_[np.full(20, .2), np.full(10, .8)][:, None]
    refs = np.r_[np.zeros(20), np.ones(10)].astype(bool)[:, None]
    persistence = np.ones_like(scores)
    rules = lr.fit_rules(scores, persistence, refs, ["a", "b"]*15, ["sound"], "adaptive")
    selected = lr.apply_rules(scores, persistence, rules)
    assert selected[:, 0].tolist() == [False]*20+[True]*10
    result = lr.metrics(selected, scores >= .1, refs)
    assert result["claims"] == result["confirmed"] == 10
    assert result["confirmed_dropped"] == 0 and result["confirmed_retained"] == 1
    zero = lr.metrics(np.zeros_like(selected), scores >= .1, refs)
    assert zero["confirmation"] is None and zero["confirmed_dropped"] == 10
    # Poisoned TEST confirmations at .7 would lower a leaked fit below .8.
    train = np.array(["dev"]*30+["test"]*10)
    extended = np.r_[scores[:, 0], np.full(10, .7)][:, None]
    extended_refs = np.r_[refs[:, 0], np.ones(10, dtype=bool)][:, None]
    assert hasattr(lr, "fit_on_dev"), "the experiment must enforce DEV-only fitting"
    again = lr.fit_on_dev(extended, np.ones_like(extended), extended_refs,
                         ["a", "b"]*15+["test_pack"]*10, ["sound"], train, "adaptive")
    assert not lr.apply_rules(np.array([[.75]]), np.ones((1, 1)), again)[0, 0], "TEST confirmations leaked into fitted threshold"
    assert lr.apply_rules(scores, persistence, again).tolist() == selected.tolist()




def label_cohorts(api):
    # Transferring AST thresholds to CED, voting approximate mappings, or silently
    # accepting a stale vocabulary must change these concrete decisions.
    import label_rules as lr
    vocab = [dict(tag="click", audioset_labels="Clicking", mapping_relation="direct", name_synonyms="click"),
             dict(tag="ui", audioset_labels="Clicking", mapping_relation="approximate", name_synonyms="ui")]
    rows = [dict(tagger=t, top_labels="Clicking 0.4", timeline="0-10: Clicking 0.4")
            for t in ("ast", "ced-base", "new-unknown")]
    rules = lambda threshold: [dict(tag="click", threshold=threshold, min_persistence=0),
                               dict(tag="ui", threshold=.1, min_persistence=0)]
    bundle = dict(version=lr.VERSION, recipe="fixture", vocabulary_signature=lr.vocabulary_signature(vocab),
                  cohorts={"ast": rules(.3), "ced-base": rules(.7)})
    got, meta = lr.decisions_for_rows(rows, vocab, bundle)
    assert got.tolist() == [[True, False], [False, False], [False, False]]
    assert meta["unknown_taggers"] == ["new-unknown"]
    vocab[0]["audioset_labels"] = "Rain"
    try:
        lr.decisions_for_rows(rows, vocab, bundle)
    except ValueError:
        pass
    else:
        assert False, "stale policy must not silently reuse a changed mapping"


def run(api):
    for test in (label_thresholds, label_windows, label_split_isolation, label_fit_and_cost, label_cohorts):
        test(api)
        print(test.__name__ + ": ok")

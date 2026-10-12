"""DEV-fitted label decisions. No inference, decoding, or library writes."""
import hashlib
import json
import math
import re
import os
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_thread_env] = "2"
from collections import Counter
import numpy as np

VERSION = "dev-label-consistency-v2"
RETENTION = .90
RECIPES = ("flat", "knee", "domain_q25", "domain_q50", "domain_q75",
           "persistence25", "persistence50", "adaptive")
PAIR = re.compile(r"(.+?) (\d+(?:\.\d+)?)(?:; |, |$)")
SPAN = re.compile(r"^\s*(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?):\s*(.*)$")


def group_key(pack):
    return re.sub(r"\s*\(\d+\)\s*$", "", pack.replace("\\", "/").casefold()).strip()


def assert_pack_isolation(packs, splits):
    groups = {}
    assert len(packs) == len(splits)
    for pack, split in zip(packs, splits):
        assert split in ("dev", "test"), "invalid split"
        key = group_key(pack)
        assert key not in groups or groups[key] == split, "DEV/TEST mix pack: " + key
        groups[key] = split


def pack_splits(packs, fixed=None, seed=20261010):
    fixed = fixed or {}
    choices = {}
    for pack in packs:
        key = group_key(pack)
        score = int(hashlib.sha256((str(seed)+":"+key).encode()).hexdigest()[:16], 16) / 2**64
        choices[key] = fixed.get(key, "dev" if score < .6 else "test")
    result = np.array([choices[group_key(p)] for p in packs])
    assert_pack_isolation(packs, result)
    return result


def label_mapping(vocabulary, relation="direct"):
    mapping = {}
    for j, entry in enumerate(vocabulary):
        if entry["mapping_relation"] != relation:
            continue
        for label in entry.get("audioset_labels", "").split(";"):
            if label.strip():
                mapping.setdefault(label.strip(), []).append(j)
    return mapping


def parsed_scores(text):
    parsed = {}
    for label, value in PAIR.findall((text or "").strip()):
        parsed[label.strip()] = max(parsed.get(label.strip(), 0), float(value))
    return parsed


def window_counts(timeline, vocabulary, mapping=None):
    """Printed top-three occupancy is a lower bound; duplicate spans count once."""
    mapping = label_mapping(vocabulary) if mapping is None else mapping
    windows = {}
    for part in (timeline or "").split("|"):
        match = SPAN.match(part)
        if not match:
            continue
        start, end = float(match[1]), float(match[2])
        if end <= start:
            continue
        tags = windows.setdefault((start, end), set())
        for label, score in parsed_scores(match[3]).items():
            if score >= .1:
                tags.update(mapping.get(label, ()))
    counts = np.zeros(len(vocabulary), dtype=np.int32)
    for tags in windows.values():
        for j in tags:
            counts[j] += 1
    return counts, len(windows)


def label_features(rows, vocabulary):
    scores = np.full((len(rows), len(vocabulary)), np.nan, dtype=np.float32)
    persistence = np.full_like(scores, np.nan)
    windows = np.zeros(len(rows), dtype=np.int16)
    mapping = label_mapping(vocabulary)
    for i, row in enumerate(rows):
        for label, value in parsed_scores(row.get("top_labels", "")).items():
            for j in mapping.get(label, ()):
                scores[i, j] = max(value, scores[i, j]) if np.isfinite(scores[i, j]) else value
        counts, total = window_counts(row.get("timeline", ""), vocabulary, mapping)
        windows[i] = total
        if total:
            persistence[i] = counts / total
    return scores, persistence, windows


def apply_rules(scores, persistence, rules):
    assert scores.shape == persistence.shape and scores.shape[1] == len(rules)
    supports = np.zeros(scores.shape, dtype=bool)
    for j, rule in enumerate(rules):
        threshold = rule["threshold"]
        if threshold is None:
            continue
        supports[:, j] = np.isfinite(scores[:, j]) & (scores[:, j] >= threshold)
        floor = rule["min_persistence"]
        if floor > 0:
            supports[:, j] &= np.isnan(persistence[:, j]) | (persistence[:, j] >= floor)
    return supports


def metrics(selected, baseline, reference):
    claims = int(selected.sum())
    confirmed = int((selected & reference).sum())
    before = int((baseline & reference).sum())
    dropped = int((baseline & reference & ~selected).sum())
    return dict(claims=claims, confirmed=confirmed,
                confirmation=confirmed/claims if claims else None,
                baseline_confirmed=before, confirmed_dropped=dropped,
                confirmed_retained=(before-dropped)/before if before else None,
                claims_dropped=int((baseline & ~selected).sum()),
                reference_claims=int(reference.sum()))


def score_knee(values):
    values = np.sort(values.astype(np.float64))[::-1]
    if len(values) < 20 or values[0] - values[-1] <= 1e-8:
        return .1
    x = np.linspace(0, 1, len(values))
    y = (values-values[-1])/(values[0]-values[-1])
    distance = (1-x)-y
    rank = int(np.argmax(distance[1:-1]))+1
    return float(values[rank]) if distance[rank] > .05 else .1


def domain_cell(row):
    try:
        duration = float(row.get("duration_seconds", ""))
    except (ValueError, TypeError):
        duration = float("nan")
    band = "unknown" if not math.isfinite(duration) else ("<=2" if duration <= 2 else "<=10" if duration <= 10 else "<=60" if duration <= 60 else ">60")
    return (row.get("bucket") or "unknown")+":"+band


def fit_rules(scores, persistence, reference, packs, tags, recipe, domains=None):
    """Caller supplies DEV arrays only. Never accepts TEST data or TEST masks."""
    if recipe not in RECIPES:
        raise ValueError("unknown recipe: " + recipe)
    assert scores.shape == persistence.shape == reference.shape
    assert len(packs) == scores.shape[0]
    domains = np.array(domains if domains is not None else ["unknown"]*len(packs))
    rules = []
    for j, tag in enumerate(tags):
        valid = np.isfinite(scores[:, j]) & (scores[:, j] >= .1)
        values = scores[valid, j]
        confirmed = valid & reference[:, j]
        n, c = int(valid.sum()), int(confirmed.sum())
        refpacks = len({group_key(packs[i]) for i in np.flatnonzero(confirmed)})
        rule = dict(tag=tag, threshold=.1, min_persistence=0., status="flat_control",
                    dev_candidates=n, dev_confirmed=c, dev_confirmed_packs=refpacks)
        if recipe == "flat":
            pass
        elif n < 20 or c < 5 or refpacks < 2:
            rule["status"] = "sparse_uncalibrated_flat_fallback"
        elif recipe == "knee":
            rule.update(threshold=score_knee(values), status="dev_score_knee")
        elif recipe.startswith("domain_q"):
            cell_counts = Counter(domains[reference[:, j]])
            cells = sorted(k for k, count in cell_counts.items() if count >= 3)
            eligible = valid & np.isin(domains, cells)
            conditional = scores[eligible, j] if int(eligible.sum()) >= 20 else values
            q = int(recipe.removeprefix("domain_q"))/100
            rule.update(threshold=float(np.quantile(conditional, q)), status="dev_domain_quantile",
                        compatible_cells=cells, quantile=q, quantile_candidates=len(conditional))
        elif recipe.startswith("persistence"):
            rule.update(min_persistence=int(recipe.removeprefix("persistence"))/100,
                        status="dev_persistence")
        else:
            # Maximize DEV confirmation with >=90% of each tag's confirmed claims retained.
            # Ties prefer no persistence gate, then the lower threshold.
            best, chosen = None, (.1, 0.)
            for floor in (0., .25, .5, .75):
                for threshold in np.unique(np.r_[np.float32(.1), values]):
                    keep = valid & (scores[:, j] >= threshold)
                    if floor:
                        keep &= np.isnan(persistence[:, j]) | (persistence[:, j] >= floor)
                    kept_c, kept_n = int((keep & reference[:, j]).sum()), int(keep.sum())
                    if not kept_n or kept_c < math.ceil(RETENTION*c-1e-8):
                        continue
                    objective = (kept_c/kept_n, -floor, -float(threshold))
                    if best is None or objective > best:
                        best, chosen = objective, (float(threshold), floor)
            rule.update(threshold=chosen[0], min_persistence=chosen[1], status="dev_consistency_optimized")
        rules.append(rule)
    return rules



def fit_on_dev(scores, persistence, reference, packs, tags, splits, recipe, domains=None):
    """Enforce pack isolation at the boundary that selects all fitting arrays."""
    assert_pack_isolation(packs, splits)
    dev = np.asarray(splits) == "dev"
    if not dev.any():
        raise ValueError("No DEV files")
    devpacks = [packs[i] for i in np.flatnonzero(dev)]
    devdomains = np.asarray(domains)[dev] if domains is not None else None
    return fit_rules(scores[dev], persistence[dev], reference[dev], devpacks, tags, recipe, devdomains)


def vocabulary_signature(vocabulary):
    values = [{k: entry[k] for k in ("tag", "audioset_labels", "mapping_relation", "name_synonyms")} for entry in vocabulary]
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def decisions_for_rows(rows, vocabulary, bundle):
    if bundle.get("version") != VERSION or bundle.get("vocabulary_signature") != vocabulary_signature(vocabulary):
        raise ValueError("Label policy version/vocabulary changed; refit before use")
    scores, persistence, _ = label_features(rows, vocabulary)
    supports = np.zeros(scores.shape, dtype=bool)
    cohort_counts = Counter()
    cohorts = {}
    for i, row in enumerate(rows):
        if row:
            cohorts.setdefault(row.get("tagger") or "legacy-unspecified", []).append(i)
    for cohort, members in cohorts.items():
        cohort_counts[cohort] = len(members)
        rules = bundle["cohorts"].get(cohort)
        if rules is None:
            continue  # Unknown tagger abstains instead of transferring uncalibrated scores.
        if [r["tag"] for r in rules] != [v["tag"] for v in vocabulary]:
            raise ValueError("Label policy tag order mismatch")
        supports[members] = apply_rules(scores[members], persistence[members], rules)
    return supports, dict(version=VERSION, recipe=bundle["recipe"], cohorts=dict(cohort_counts),
                          unknown_taggers=sorted(set(cohorts)-set(bundle["cohorts"])),
                          policy_training="DEV packs only; frozen for TEST and library use",
                          persistence_note="Printed top-three per-window occupancy is a lower bound; unavailable timeline passes the persistence gate.")

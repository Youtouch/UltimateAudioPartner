"""Cross-reference canonical tags without decoding audio or treating missing evidence as absence.

Examples:
  python vocab_tags.py build --root <library> --out <tagging>/AUDIO_VOCAB_TAGS.csv
  python vocab_tags.py disagreements --root <library> --out <tagging>/AUDIO_TAG_REVIEW.csv
  python vocab_tags.py stats --root <library> --out <tagging>/AUDIO_VOCAB_STATS.csv
  python vocab_tags.py check

Default outputs follow the CLI contract (in --root). Use --out to keep a library read-only.
All caches and temporary self-test fixtures stay beside this script.
"""
import os
import sys

sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import csv
import hashlib
import io
import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SKILL = HERE.parent
sys.path.insert(0, str(SKILL / "scripts"))
# Reuse the actual search implementation; Index is deliberately avoided because it writes name caches.
from audio_search import clean_name, words, stem, PAIR, DIM

MODEL = "laion/clap-htsat-unfused"
RULE_VERSION = "upper-tail-knee-v1"
SOURCES = ("name", "label", "audio")
SILENCE_TAG = "silence_or_room_tone"
NEGATIONS = {"no", "not", "without", "excluding", "except"}
TAG_FIELDS = ["relative_path", "tag", "name", "label", "audio", "sources", "tagger", "built_at"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_csv_snapshot(path, optional=False):
    """One open/read pins the old or new CSV across a rename; incomplete last records are ignored."""
    if optional and not path.exists():
        return [], {"missing": True}
    raw = path.read_bytes()
    boundary = raw.rfind(b"\n") + 1
    complete = raw if raw.endswith(b"\n") else raw[:boundary]
    reader = csv.DictReader(io.StringIO(complete.decode("utf-8-sig")), strict=True)
    if not reader.fieldnames or "relative_path" not in reader.fieldnames:
        raise ValueError(f"{path.name}: missing relative_path header")
    result = []
    for line, row in enumerate(reader, 2):
        if None in row or any(v is None for v in row.values()) or not row["relative_path"]:
            raise ValueError(f"{path.name}: malformed complete record {line}")
        result.append(row)
    return result, {"bytes_read": len(raw), "complete_bytes": len(complete),
                    "tail_bytes_ignored": len(raw) - len(complete),
                    "sha256": digest(complete), "rows": len(result), "columns": reader.fieldnames}


@dataclass
class Library:
    paths: list
    packs: list
    rows: list
    audio: np.ndarray
    valid_audio: np.ndarray
    silence: np.ndarray
    meta: dict


def numeric(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def load_library(root):
    root = Path(root)
    tagged_rows, tag_meta = read_csv_snapshot(root / "AUDIO_TAGS.csv", optional=True)
    ledger_rows, ledger_meta = read_csv_snapshot(root / "AUDIO_LEDGER_files.csv", optional=True)
    tagged = {r["relative_path"]: r for r in tagged_rows}  # latest, even if its vector is unusable
    ledger = {r["relative_path"]: r for r in ledger_rows}
    paths = sorted(set(tagged) | set(ledger))
    rows = [tagged.get(p, {}) for p in paths]
    packs = [ledger.get(p, {}).get("pack") or (p.replace("\\", "/").split("/")[0] if "/" in p.replace("\\", "/") else "") for p in paths]
    vectors = root / "AUDIO_TAGS_clap.f32"
    size = vectors.stat().st_size if vectors.exists() else 0
    vector_count, tail = divmod(size, DIM * 4)
    indices = np.full(len(paths), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        try:
            indices[i] = int(row.get("clap_row", ""))
        except (ValueError, TypeError):
            pass
    valid = (indices >= 0) & (indices < vector_count)
    audio = np.zeros((len(paths), DIM), dtype=np.float32)
    if vector_count:
        mapped = np.memmap(vectors, mode="r", dtype="<f4", shape=(vector_count, DIM))
        audio[valid] = mapped[indices[valid]]
        del mapped
        norms = np.linalg.norm(audio, axis=1)
        valid &= np.isfinite(audio).all(axis=1) & (norms > 0)
        audio[valid] /= norms[valid, None]
        audio[~valid] = 0
    silence = np.array([numeric(r.get("silence")) for r in rows], dtype=np.float32)
    clap_metadata = {}
    if (root / "AUDIO_TAGS_clap.json").exists():
        clap_metadata = json.loads((root / "AUDIO_TAGS_clap.json").read_text(encoding="utf-8-sig"))
        if clap_metadata.get("model", MODEL) != MODEL:
            raise ValueError("Audio vectors use a different model than the canonical prompts")
    taggers = Counter(r.get("tagger") or "legacy-unspecified" for r in tagged.values())
    meta = {"root": str(root.resolve()), "tag_csv": tag_meta, "ledger_csv": ledger_meta,
            "files": len(paths), "tagged_files": len(tagged), "ledger_files": len(ledger),
            "duplicate_tag_rows": len(tagged_rows) - len(tagged),
            "ledger_only_files": len(set(ledger) - set(tagged)), "valid_audio_files": int(valid.sum()),
            "invalid_or_missing_audio_files": int((~valid).sum()),
            "vector_rows": vector_count, "vector_bytes_observed": size, "vector_tail_bytes_ignored": tail,
            "taggers": dict(taggers), "clap_metadata": clap_metadata,
            "relabel_in_progress_at_load": (root / "AUDIO_TAGS.relabel.csv").exists(),
            "legacy_tagger_note": "Missing tagger is recorded as legacy-unspecified; AST is inferred only from the supplied history.",
            "label_scope": "Printed top 8 >=0.1 only; missing labels are unknown, not negative.",
            "audio_scope": "Cached first 10 seconds; no audio decoded or heard.",
            "name_vector_cache": "Not used: name support is lexical, preserving its separate evidence mechanism."}
    return Library(paths, packs, rows, audio, valid, silence, meta)


def load_vocabulary(path=None):
    canonical = list(csv.DictReader((SKILL / "bench" / "vocabulary.csv").open(encoding="utf-8-sig", newline="")))
    synonym_path = Path(path) if path else HERE / "vocabulary_synonyms.csv"
    with synonym_path.open(encoding="utf-8-sig", newline="") as f:
        synonym_rows = list(csv.DictReader(f))
    synonyms = {r["tag"]: r for r in synonym_rows}
    if len(synonyms) != len(synonym_rows) or set(synonyms) != {r["tag"] for r in canonical}:
        raise ValueError("Synonym vocabulary must contain each canonical tag exactly once")
    out = []
    for original in canonical:
        row = synonyms[original["tag"]]
        prompts = [row.get(f"prompt_{i}", "").strip() for i in (1, 2, 3) if row.get(f"prompt_{i}", "").strip()]
        if not 1 <= len(prompts) <= 3 or not row.get("name_synonyms", "").strip():
            raise ValueError(f"Incomplete synonym/prompt entry: {original['tag']}")
        out.append({**original, "name_synonyms": row["name_synonyms"], "prompts": prompts})
    return out


def name_tokens(relative_path, pack):
    relative_path = relative_path.replace("\\", "/")
    # clean_name expects a first pack directory; a neutral virtual directory preserves root filenames.
    if "/" not in relative_path:
        relative_path = "__catalogue__/" + relative_path
    return tuple(clean_name(relative_path, pack).split())


def synonym_patterns(entry):
    return [tuple(stem(w) for w in words(s)) for s in entry["name_synonyms"].split(";") if words(s)]


def token_match(tokens, patterns):
    normalized = tuple(stem(w) for w in tokens)
    for pattern in patterns:
        width = len(pattern)
        for start in range(len(tokens) - width + 1):
            if normalized[start:start + width] == pattern:
                if not set(tokens[max(0, start - 2):start]) & NEGATIONS:
                    return True
    return False


def match_name(relative_path, pack, entry):
    return token_match(name_tokens(relative_path, pack), synonym_patterns(entry))


def map_labels(top_labels, vocabulary):
    # PAIR handles commas inside AudioSet names. Never sum mapped parent and child scores.
    parsed = {}
    for label, score in PAIR.findall(top_labels or ""):
        value = numeric(score)
        if math.isfinite(value):
            parsed[label.strip()] = max(parsed.get(label.strip(), -math.inf), value)
    direct = np.full(len(vocabulary), np.nan, dtype=np.float32)
    weak = np.full(len(vocabulary), np.nan, dtype=np.float32)
    for j, tag in enumerate(vocabulary):
        values = [parsed[label.strip()] for label in tag.get("audioset_labels", "").split(";") if label.strip() in parsed]
        if values:
            relation = tag["mapping_relation"]
            if relation == "direct":
                direct[j] = max(values)
            elif relation == "approximate":
                weak[j] = max(values)
    return direct, weak


def tag_audio_decisions(scores, valid_audio, silence, tag_names, top_k=None):
    """Per-tag upper-tail geometric knee AND stable top K; no universal cosine cutoff."""
    supports = np.zeros(scores.shape, dtype=bool)
    rules = []
    for j, tag in enumerate(tag_names):
        eligible = np.asarray(valid_audio, dtype=bool) & np.isfinite(scores[:, j])
        if tag != SILENCE_TAG:
            eligible &= ~(np.asarray(silence) >= 0.5)
        ids = np.flatnonzero(eligible)
        order = ids[np.argsort(-scores[ids, j], kind="stable")]
        n = len(order)
        cap = int(top_k) if top_k is not None else min(1000, max(10, math.ceil(0.05 * n)))
        if cap < 1:
            raise ValueError("top_k must be positive")
        rule = {"tag": tag, "rule_version": RULE_VERSION, "eligible": n, "top_k": min(cap, n),
                "tail_fraction": 0.10, "tail_size": 0, "knee_score": None,
                "knee_rank": None, "knee_prominence": None, "selected": 0, "status": "insufficient_scores"}
        if n >= 5:
            tail_size = min(n, max(5, math.ceil(0.10 * n)))
            tail_values = scores[order[:tail_size], j].astype(np.float64)
            spread = float(tail_values[0] - tail_values[-1])
            rule["tail_size"] = tail_size
            if spread > 1e-7:
                x = np.linspace(0.0, 1.0, tail_size)
                y = (tail_values - tail_values[-1]) / spread
                distance = (1.0 - x) - y
                knee = int(np.argmax(distance))
                prominence = float(distance[knee])
                rule["knee_prominence"] = prominence
                if prominence > 0.05 and 0 < knee < tail_size - 1:
                    threshold = float(tail_values[knee])
                    # Inclusive observed boundary, with stable path-order tie breaking at the rank cap.
                    selected = order[:min(cap, n)]
                    selected = selected[scores[selected, j] >= threshold]
                    supports[selected, j] = True
                    rule.update(knee_score=threshold, knee_rank=knee + 1,
                                selected=len(selected), status="candidate_support")
                else:
                    rule["status"] = "no_distinct_knee"
            else:
                rule["status"] = "flat_tail"
        rules.append(rule)
    return supports, rules


def cached_checkpoint(explicit=None):
    if explicit:
        candidates = [Path(explicit)]
    else:
        from huggingface_hub.constants import HF_HUB_CACHE
        snapshots = Path(HF_HUB_CACHE) / ("models--" + MODEL.replace("/", "--")) / "snapshots"
        candidates = sorted(snapshots.glob("*"))
    for candidate in candidates:
        if ((candidate / "config.json").exists()
                and ((candidate / "model.safetensors").exists() or (candidate / "pytorch_model.bin").exists())
                and (candidate / "tokenizer_config.json").exists()):
            return candidate.resolve()
    raise FileNotFoundError("No complete local CLAP checkpoint; supply --cached-model. Downloads are disabled.")


def prompt_vectors(vocabulary, explicit_model=None):
    checkpoint = cached_checkpoint(explicit_model)
    prompts = [p for tag in vocabulary for p in tag["prompts"]]
    provenance = {"model": MODEL, "snapshot": checkpoint.name, "checkpoint": str(checkpoint),
                  "config_sha256": digest((checkpoint / "config.json").read_bytes()),
                  "prompts_sha256": digest(json.dumps(prompts, ensure_ascii=False).encode()),
                  "aggregation": "mean of normalized prompt embeddings, then unit-normalize"}
    key = digest(json.dumps(provenance, sort_keys=True).encode())
    cache = HERE / "prompt_vectors.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["cache_key"].item()) == key:
                vectors = data["vectors"].copy()
                if vectors.shape == (len(vocabulary), DIM) and np.isfinite(vectors).all() and np.all(np.linalg.norm(vectors, axis=1) > 0):
                    return vectors, {**provenance, "cache_hit": True}
    import torch
    import transformers
    from transformers import AutoTokenizer, ClapTextModelWithProjection
    from audio_tag import embed_text, _models
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    transformers.logging.set_verbosity_error()  # Full checkpoint's unused audio tower is expected.
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model, loading = ClapTextModelWithProjection.from_pretrained(
        str(checkpoint), local_files_only=True, use_safetensors=(checkpoint / "model.safetensors").exists(),
        output_loading_info=True)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise ValueError(f"Text checkpoint did not load completely: {loading}")
    model = model.to("cpu").eval()

    class TextTowerAdapter:
        device = torch.device("cpu")

        def get_text_features(self, **inputs):
            return model(**inputs).text_embeds

    # embed_text is reused unchanged; its normal clap() resolver now sees our local CPU-only text tower.
    _models["clap"] = (tokenizer, TextTowerAdapter())
    vectors = []
    for begin in range(0, len(prompts), 16):
        vectors.extend(embed_text(prompts[begin:begin + 16]))
        print(f"CPU text prompts {min(begin + 16, len(prompts))}/{len(prompts)}", flush=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    aggregated = []
    begin = 0
    for tag in vocabulary:
        count = len(tag["prompts"])
        vector = vectors[begin:begin + count].mean(axis=0)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm <= 0:
            raise ValueError("Invalid CLAP text embedding")
        aggregated.append(vector / norm)
        begin += count
    aggregated = np.asarray(aggregated, dtype=np.float32)
    np.savez_compressed(cache, cache_key=np.array(key), vectors=aggregated)
    return aggregated, {**provenance, "cache_hit": False, "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": "cpu", "threads": 2}


def source_statistics(support, tag_names):
    result = []
    for j, tag in enumerate(tag_names):
        present = support[:, j, :]
        counts = present.sum(axis=1)
        row = {"tag": tag, **{f"files_{n}": int((counts == n).sum()) for n in (1, 2, 3)}}
        for a, source in enumerate(SOURCES):
            positives = present[:, a]
            denom = int(positives.sum())
            row[f"{source}_files"] = denom
            for b, other in enumerate(SOURCES):
                if a == b:
                    continue
                intersection = int((positives & present[:, b]).sum())
                row[f"{source}_to_{other}"] = intersection / denom if denom else None
                row[f"{source}_to_{other}_confirmed"] = intersection
                row[f"{source}_to_{other}_denominator"] = denom
            others = present[:, [b for b in range(3) if a != b]].any(axis=1)
            intersection = int((positives & others).sum())
            other_denom = int(others.sum())
            row[f"{source}_loo_precision"] = intersection / denom if denom else None
            row[f"{source}_loo_recall"] = intersection / other_denom if other_denom else None
            row[f"{source}_loo_confirmed"] = intersection
            row[f"{source}_loo_other_union"] = other_denom
        result.append(row)
    return result


def write_csv(path, rows, fields):
    path = Path(path)
    if not path.parent.exists():
        raise FileNotFoundError(f"Output folder does not exist: {path.parent}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tagger_name(row):
    return row.get("tagger") or ("legacy-unspecified" if row else "not-evaluated")


def build_evidence(library, vocabulary, model=None, top_k=None):
    n, t = len(library.paths), len(vocabulary)
    name = np.zeros((n, t), dtype=bool)
    labels = np.full((n, t), np.nan, dtype=np.float32)
    weak = np.full((n, t), np.nan, dtype=np.float32)
    patterns = [synonym_patterns(tag) for tag in vocabulary]
    for i, (path, pack, row) in enumerate(zip(library.paths, library.packs, library.rows)):
        tokens = name_tokens(path, pack)
        name[i] = [token_match(tokens, p) for p in patterns]
        labels[i], weak[i] = map_labels(row.get("top_labels", ""), vocabulary)
    text, text_meta = prompt_vectors(vocabulary, model)
    scores = library.audio @ text.T
    scores[~library.valid_audio] = np.nan
    audio, rules = tag_audio_decisions(scores, library.valid_audio, library.silence,
                                       [v["tag"] for v in vocabulary], top_k)
    support = np.stack((name, np.isfinite(labels), audio), axis=2)
    return support, labels, weak, scores, rules, text_meta


def evidence_rows(library, vocabulary, support, labels, scores, built_at):
    for i, j in np.argwhere(support.any(axis=2)):
        yield {"relative_path": library.paths[i], "tag": vocabulary[j]["tag"],
               "name": int(support[i, j, 0]), "label": float(labels[i, j]) if support[i, j, 1] else "",
               "audio": float(scores[i, j]) if support[i, j, 2] else "",
               "sources": int(support[i, j].sum()), "tagger": tagger_name(library.rows[i]), "built_at": built_at}


def percentile_ranks(values):
    result = np.full(len(values), np.nan)
    ids = np.flatnonzero(np.isfinite(values))
    if not len(ids):
        return result
    ordered = np.sort(values[ids])
    result[ids] = np.searchsorted(ordered, values[ids], side="right") / len(ids)
    return result


def pack_key(pack):
    return re.sub(r"\s*\(\d+\)\s*$", "", pack.strip().lower())


def review_candidates(library, vocabulary, support, scores, labels, rules, top=150):
    """Priority is a review scheduling heuristic, never a probability or a truth judgement."""
    if top < 1:
        raise ValueError("top must be positive")
    tags = [v["tag"] for v in vocabulary]
    indices = {tag: j for j, tag in enumerate(tags)}
    counts = support.sum(axis=2)
    voice = {"speech", "whisper", "sung_voice", "choir", "chant", "humming"}
    ranked = []
    for j, tag in enumerate(tags):
        audio_rank = percentile_ranks(scores[:, j])
        label_rank = np.full(len(library.paths), np.nan)
        # Uncalibrated classifier scores are compared only within the same tagger cohort.
        cohorts = {}
        for i, row in enumerate(library.rows):
            cohorts.setdefault(tagger_name(row), []).append(i)
        for members in cohorts.values():
            members = np.asarray(members, dtype=np.int64)
            label_rank[members] = percentile_ranks(labels[members, j])
        rarity = 1.0 / math.sqrt(max(1, int((counts[:, j] > 0).sum())))
        for i in np.flatnonzero(counts[:, j] == 1):
            src = int(np.flatnonzero(support[i, j])[0])
            strength = 1.0 if src == 0 else float(label_rank[i] if src == 1 else audio_rank[i])
            if src != 0 and (not math.isfinite(strength) or strength < 0.90):
                continue
            other_states = []
            if src != 0:
                other_states.append("name has no matching synonym")
            if src != 1:
                other_states.append("label has no printed direct match (top-8 absence is unknown)"
                                    if library.rows[i] else "labels not evaluated")
            if src != 2:
                if not library.valid_audio[i]:
                    other_states.append("audio not evaluated: missing/invalid vector")
                elif tag != SILENCE_TAG and library.silence[i] >= 0.5:
                    other_states.append("audio support blocked by silence guard")
                else:
                    other_states.append("audio does not pass this tag's knee/rank rule")
            tensions = []
            # Only the explicitly negative intended name instrumental raises a voice-presence question.
            if tag == "instrumental" and src == 0:
                rivals = [v for v in voice if v in indices and support[i, indices[v], 1:].any()]
                if rivals:
                    tensions.append("instrumental name versus model vocal evidence: " + "/".join(sorted(rivals)))
            if src in (1, 2) and tag in voice and "instrumental" in indices and support[i, indices["instrumental"], 0]:
                tensions.append("model vocal claim versus instrumental filename")
            if src in (1, 2) and tag in voice and "calm" in indices and support[i, indices["calm"], 0]:
                tensions.append("vocal-layer question under calm-named material (can coexist)")
            evaluated = int(bool(library.rows[i])) + int(library.valid_audio[i])
            priority = 3.0 * bool(tensions) + 2.0 * strength + rarity + 0.1 * evaluated
            value = "exact synonym" if src == 0 else (
                f"printed direct label={float(labels[i, j]):.4g}" if src == 1
                else f"audio cosine={float(scores[i, j]):.4g}, within-tag percentile={strength:.3f}")
            reason = f"{SOURCES[src]} alone supports {tag} ({value}); " + "; ".join(other_states + tensions)
            ranked.append({"relative_path": library.paths[i], "tag": tag, "source": SOURCES[src],
                           "pack": library.packs[i], "priority": priority, "reason": reason,
                           "tagger": tagger_name(library.rows[i])})
    ranked.sort(key=lambda r: (-r["priority"], r["relative_path"], r["tag"]))
    chosen, per_tag, per_pack, seen, families = [], Counter(), Counter(), set(), set()
    for row in ranked:
        path, tag, pack = row["relative_path"], row["tag"], pack_key(row["pack"])
        parts = path.replace("\\", "/").rsplit("/", 1)
        filename = parts[-1].rsplit(".", 1)[0]
        parent_parts = path.replace("\\", "/").split("/")[:-1]
        normalized_parent = "/".join(" ".join(words(part)) for part in parent_parts if words(part))
        family = (pack, normalized_parent, " ".join(words(filename)))
        if path in seen or family in families or per_tag[tag] >= 10 or per_pack[pack] >= 5:
            continue
        chosen.append({"rank": len(chosen) + 1, **row})
        seen.add(path)
        families.add(family)
        per_tag[tag] += 1
        per_pack[pack] += 1
        if len(chosen) >= top:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="assert-based synthetic tests; no model or library needed")
    for command in ("build", "disagreements", "stats"):
        p = sub.add_parser(command)
        p.add_argument("--root", type=Path, default=os.environ.get("AUDIO_LIBRARY"))
        p.add_argument("--out", type=Path)
        p.add_argument("--cached-model", type=Path, help="complete local cached CLAP checkpoint")
        p.add_argument("--audio-top-k", type=int, default=None)
        if command == "disagreements":
            p.add_argument("--top", type=int, default=150)
    args = parser.parse_args()
    if args.cmd == "check":
        if not __debug__:
            parser.error("Self-tests require asserts: do not use python -O")
        from self_tests import run
        return run()
    if not args.root:
        parser.error("--root or AUDIO_LIBRARY is required")
    if args.audio_top_k is not None and args.audio_top_k < 1:
        parser.error("--audio-top-k must be positive")
    start = time.perf_counter()
    built_at = datetime.now(timezone.utc).isoformat()
    library = load_library(args.root)
    vocabulary = load_vocabulary()
    print(f"{len(library.paths)} files; {int(library.valid_audio.sum())} valid cached audio vectors; CPU only", flush=True)
    support, labels, weak, scores, rules, text_meta = build_evidence(library, vocabulary, args.cached_model, args.audio_top_k)
    defaults = {"build": "AUDIO_VOCAB_TAGS.csv", "disagreements": "AUDIO_TAG_REVIEW.csv", "stats": "AUDIO_VOCAB_STATS.csv"}
    out = args.out or args.root / defaults[args.cmd]
    tags = [v["tag"] for v in vocabulary]
    stats = source_statistics(support, tags)
    count = support.sum(axis=2)
    if args.cmd == "build":
        write_csv(out, evidence_rows(library, vocabulary, support, labels, scores, built_at), TAG_FIELDS)
        weak_rows = ({"relative_path": library.paths[i], "tag": tags[j], "weak_label_score": float(weak[i, j]),
                      "mapped_labels": vocabulary[j]["audioset_labels"], "mapping_relation": "approximate",
                      "tagger": tagger_name(library.rows[i])} for i, j in np.argwhere(np.isfinite(weak)))
        write_csv(out.parent / "AUDIO_VOCAB_WEAK.csv", weak_rows,
                  ["relative_path", "tag", "weak_label_score", "mapped_labels", "mapping_relation", "tagger"])
        coverage = ({"relative_path": path, "pack": library.packs[i],
                     "tagger": tagger_name(library.rows[i]), "labels_evaluated": int(bool(library.rows[i])),
                     "audio_evaluated": int(library.valid_audio[i]), "silence": library.rows[i].get("silence", ""),
                     "tags_supported": int((count[i] > 0).sum()),
                     "name_tags": int(support[i, :, 0].sum()), "label_tags": int(support[i, :, 1].sum()),
                     "audio_tags": int(support[i, :, 2].sum())} for i, path in enumerate(library.paths))
        write_csv(out.parent / "AUDIO_VOCAB_COVERAGE.csv", coverage,
                  ["relative_path", "pack", "tagger", "labels_evaluated", "audio_evaluated", "silence",
                   "tags_supported", "name_tags", "label_tags", "audio_tags"])
    elif args.cmd == "disagreements":
        review = review_candidates(library, vocabulary, support, scores, labels, rules, args.top)
        write_csv(out, review, ["rank", "relative_path", "tag", "source", "pack", "priority", "reason", "tagger"])
    else:
        write_csv(out, stats, list(stats[0]))
    write_csv(out.parent / "AUDIO_VOCAB_RULES.csv", rules, list(rules[0]))
    manifest = {**library.meta, "built_at": built_at, "command": args.cmd, "output": str(out.resolve()),
                "vocabulary_tags": len(tags), "canonical_vocabulary_sha256": digest((SKILL / "bench" / "vocabulary.csv").read_bytes()),
                "synonyms_sha256": digest((HERE / "vocabulary_synonyms.csv").read_bytes()), "text_encoder": text_meta,
                "rule_version": RULE_VERSION, "audio_rules": rules, "supported_pairs": int((count > 0).sum()),
                "pairs_by_source_count": {str(k): int((count == k).sum()) for k in (1, 2, 3)},
                "files_with_tags": int((count > 0).any(axis=1).sum()),
                "files_without_supported_tags": int((~(count > 0).any(axis=1)).sum()),
                "weak_mapping_pairs": int(np.isfinite(weak).sum()),
                "silence_guard_files": int((library.silence >= 0.5).sum()),
                "cpu_thread_cap": 2, "elapsed_seconds": time.perf_counter() - start,
                "assumptions": ["Evidence mechanisms are separate, not guaranteed statistically independent.",
                                "Name matching is lexical; folder/source identity and listener judgement are unverified.",
                                "Direct printed labels count as candidate support, without calibration.",
                                "Knee, geometric prominence >0.05, 10% tail and 5%/1000 rank cap are unvalidated screening heuristics.",
                                "Missing silence measurement leaves guard unknown; it is never inferred from absence of labels.",
                                "Tag evidence may refer to different analyzed spans; never certifies whole-file absence.",
                                "Default outputs write to root; this task's executions explicitly use --out in tagging."]}
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "files": len(library.paths), "supported_pairs": manifest["supported_pairs"],
                      "pairs_by_source_count": manifest["pairs_by_source_count"],
                      "zero_tag_files": manifest["files_without_supported_tags"], "taggers": manifest["taggers"],
                      "elapsed_seconds": round(manifest["elapsed_seconds"], 2)}, indent=2))


if __name__ == "__main__":
    main()


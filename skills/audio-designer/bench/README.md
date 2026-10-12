# Audio tagging benchmark

This is a 252-clip engineering pilot for comparing retrieval and studio source tags. All truth is initially blank. `selection_audit.csv`, exported AST labels, strata, filenames and the proxy results are candidate evidence, never listening verdicts.

## Review in about 20 seconds per clip

Open `benchmark.csv` in a spreadsheet and keep `vocabulary.csv` nearby. Preserve the path, group, split and stratum columns. `review_spans.csv` fixes every audition to **0 to min(10, file duration) seconds**. `review_playlist.m3u` provides original-file links and VLC start/stop hints; VLC support is not verified. You can instead paste the relative path into the already-running sound browser and listen only to that span. Keep a consistent comfortable playback volume.

1. Listen once (up to 10 seconds). Long tracks are deliberately excerpt reviews, not whole-file reviews.
2. In `sources`, enter all supported canonical tags separated by semicolons, for example `bird_song;wind;ambience_bed`. Use the most specific supported source; add independent layers such as `music`, `instrumental`, `single_event`, `soft` or a UI role when justified. Do not automatically add every parent label. UI roles and mood are listener judgements.
3. In `foreground_or_background`, write `tag:foreground;tag:background` for mixtures, or `foreground`, `background`, `mixed`. These notes are preserved but not scored by the current harness.
4. Set `uncertain=n` only when the sources list is a **complete review of this excerpt**. Add your initials in `reviewed_by`. Use `notes` only when helpful. If it takes longer than about 20 seconds, set `uncertain=y` and leave a short question for a later pass.

**Present:** listed in `sources` on a completed row. **Absent after review:** omitted from a row with both `uncertain=n` and nonempty `reviewed_by`; that explicit completion declaration makes omissions negatives for this excerpt only. **Uncertain:** `uncertain=y`; the entire row is excluded from reviewed metrics, even if some tags were entered. **Not reviewed:** reviewer or completion flag missing; excluded, never treated as negative. Blank sources on a completed row means none of the vocabulary applies; ordinary near silence should receive `silence_or_room_tone`. Use `unknown_source` for an audible unidentified source and `uncertain=y` if you cannot complete the source inventory. A low or unprinted model score never establishes absence.

The ten-second cap limits review burden (roughly 60-85 minutes including annotation for 252 clips). It cannot establish absence of voices or late events in a whole song or ambience. Do not make whole-file claims from this pilot. For an important disagreement, obtain a second listen and preserve its reason in notes before adjudicating.

## Files and reproducibility

- `benchmark.csv`: the only review sheet; initially five empty review columns.
- `vocabulary.csv`: 89 canonical tags, definitions, AudioSet mappings by **label name**, plus direct/approximate/studio-only mapping type. Approximate mappings are not interchangeable classifier targets. Names were validated against the cached AST configuration.
- `queries.csv`: 28 frozen queries with positive tags, any/all matching, excluded tags, duration limits, and separate filename proxy regexes. Some descriptions have unscored details (e.g. dawn, wooden/soft click); this pilot measures the listed tag contract, not all natural-language attributes.
- `selection_audit.csv`: candidate positive/hard-negative nominations, original pack, heuristic variant family and current exported AST top labels. Keep this hidden while reviewing where practical.
- `snapshot.csv` and `selection.json`: deduplicated input metadata, source byte lengths, append-tail handling, selection counts, seed and provenance. Last CSV row per path wins, including when a newer row has an unusable vector; no fallback to an older fingerprint. The current single-line tag CSV contract ignores an incomplete final record. Complete malformed rows fail closed; quoted multiline partial records are not supported. Vector tails are ignored in memory; source files are never truncated.
- `clap_baseline_scores.npz`: unrounded float32 query scores, query text vectors, paths and query IDs for the full frozen tagged snapshot. Load with `numpy.load(..., allow_pickle=False)`.
- `clap_rankings.csv`: all benchmark candidates ranked per query. `clap_library_top100.csv` and `clap_library_proxy.json`: separate full-snapshot diagnostic; do not confuse it with dev/test pilot results.
- `clap_proxy.json` and `REPORT.md`: pilot proxy metrics and limitations.

Pack IDs use the ledger pack (case normalized, trailing duplicate `(1)` etc removed). Entire packs belong to one split, stronger than keeping just track variants together. Sampling also retains at most one heuristic variant family; alternate codecs and Cut/Intensity/numbered variants are collapsed conservatively. The split is about 60/40 by pack group, balanced by stratum. No pack contributes over eight clips. This does not identify cross-pack duplicate recordings or prove pretrained models never saw these assets. A human can correct `group_id` before freezing a future benchmark version, but must then regenerate the split coherently. Do not tune against final test clips or their results; use dev for prompts and thresholds and evaluate test once choices are frozen. The current test proxy is a diagnostic requested for this initial snapshot, not a clean final accuracy result.

## Commands (PowerShell)

From this folder:

```powershell
python .\bench_select.py check
python .\bench_eval.py check
python .\bench_clap.py check
python .\bench_eval.py run --rankings .\clap_rankings.csv --mode proxy --out .\proxy_rerun.json
python .\bench_eval.py run --rankings .\clap_rankings.csv --mode reviewed --out .\reviewed_results.json
```

The reviewed command currently reports no reviewed rows. It does not silently fall back to proxy truth. Every proxy report prints **PROXY - filename truth, biased toward name-based methods**.

`bench_select.py build` refuses to overwrite an existing benchmark. For a deliberate new pilot, use `--out` with a new child folder under `bench`, and copy the frozen vocabulary/queries there. Seed defaults to 20261010. Do not regenerate the pilot merely because another tagging batch completes. `bench_clap.py run` uses only an existing local checkpoint, offline, CPU, one text at a time, and two CPU threads by default. It never re-embeds audio or writes to the library. It refuses zero/nonfinite vectors. It writes baseline outputs in this script's folder; preserve them before an intentional rerun.

## Plug in another ranking method

Supply CSV columns `query_id,relative_path,rank` (positive integer rank, unique per query); `score` is optional and ignored for ordering. A full-library ranking is accepted; paths outside a split's known benchmark rows are filtered out. Duplicate paths are collapsed at their first rank. An absent query or short result list is evaluated as incomplete and its coverage is reported.

Alternatively:

```powershell
python .\bench_eval.py run --callable .\my_method.py:rank --mode reviewed
```

A callable `rank(query_dict, candidate_rows)` returns relative path strings (or dicts with `relative_path`) in best-first order. It receives the fixed query and candidate `relative_path` / `duration_seconds` only; truth, notes, splits and nomination strata are withheld. The plugin must not read benchmark truth from disk or learn from test data. A plugin is trusted local Python, not a sandbox. Freeze the method, queries and operating threshold before test evaluation. No adapter has been run against Claude's fused search yet.

For classification, add a **full-score** CSV with `relative_path,tag,score` and `--tag-scores .\scores.csv --threshold 0.5`. Scores may use any finite scale; the operating threshold must match that scale. Convert model outputs through the named vocabulary mapping, with an explicit aggregation rule of your own. Supply every reviewed path for each tag being assessed. Incomplete per-tag score coverage yields `incomplete_scores`, not invented zero scores. Truncated AST top_labels cannot substitute for full vectors. Unsupported studio-only targets should be left incomplete and will not enter classification summaries. Proxy mode deliberately does not turn query regexes into an exhaustive 89-tag classification truth table.

## Metric definitions

Dev and test are evaluated separately. Uncertain/unreviewed rows are removed from both candidate ranks and relevance (condensed evaluation); report the judged coverage, since partial review can bias scores. This is a closed-pilot evaluation, not full-library human accuracy.

- P@10 and P@50 divide by **10 and 50**, including when fewer results are returned. Missing result slots are misses.
- AP is sum of precision at each retrieved relevant rank divided by **all known relevant** items in that split, including relevant items not returned. Queries with zero positive support get AP `null`/NA, excluded from mAP; their P@k stays zero when the judged candidate set is nonempty. Empty judged sets give NA.
- Retrieval macro is the mean over defined query scores. Micro P@k pools hits and k result slots across queries. `AP_support_weighted` is weighted by positive support; it is explicitly not a separate universal micro-AP definition.
- Per-tag precision/recall/F1 show TP/FP/FN and positive support at `score >= threshold`. Undefined zero denominators are NA. Macro averages defined values; `macro_supported` further restricts to positive-support tags. Micro pools TP/FP/FN across complete tags. No calibration or confidence intervals are claimed.

Proxy regexes match **basename only** and optional excluded basename terms, plus duration limits. They do not prove voice absence, source identity, quietness, mood or suitability. Labels and filenames were also used to nominate the pilot, so selection bias remains. Do not select a winning model from proxy numbers.

## CED trial pending explicit download approval

No packages or models were downloaded or installed. Proposed official Hugging Face checkpoints: [mispeech/ced-tiny](https://huggingface.co/mispeech/ced-tiny/tree/main), approximately **22 MB** safetensors; [mispeech/ced-base](https://huggingface.co/mispeech/ced-base/tree/main), approximately **343 MB** safetensors; plus approximately **120 kB** combined configuration and Python source. Both model cards and their Python source state **Apache-2.0**. The official feature extractor imports Torchaudio, which is absent locally. Proposed additional package: **Torchaudio 2.11.0**, CPython 3.12 Windows x86-64 wheel, **328.7 kB**, **BSD 2-Clause**, installed with `--no-deps --target bench/vendor` to preserve the running processes' environment. The existing torch 2.11.0+cu128 would remain untouched. Wheel compatibility is not yet tested; no other package installation is authorized by this proposal. Package metadata: [PyPI wheel size](https://pypi.org/project/torchaudio/2.11.0/#files), [official licence](https://github.com/pytorch/audio/blob/main/LICENSE). The separate [RicherMans/CED repository](https://github.com/RicherMans/CED) states **GPL-3.0**; that repository is not part of the proposed download. Sizes are publisher-rounded decimal values, checked October 10, 2026 Toronto date. We would omit ONNX files and retain pinned revisions and hashes in `bench`.

The optional trial will use this pilot only, batch size one, fixed excerpt provenance, CPU and GPU timings separately (decode, feature extraction and forward time distinguished), GPU synchronization and allocator peak, process peak working set, all 527 unrounded scores keyed by path in `ced_scores.npz`, and a per-clip table beside the existing AST summaries. AST summaries may cover up to 60 seconds; this mismatch must be disclosed or AST rerun on the identical spans before making a model-quality comparison. CED's current mean-pooling implementation already applies sigmoid in its head: do not apply sigmoid a second time. Runtime, memory, actual labels and compatibility remain **unverified until approval and execution**. This is a bounded experiment, not permission to retag the library or pick a winner.

References: user-supplied deep research sections 1 and 6.1-6.4; installed audio-designer SKILL.md and recognise-and-select.md. Listening truth always outranks a filename or model reading.

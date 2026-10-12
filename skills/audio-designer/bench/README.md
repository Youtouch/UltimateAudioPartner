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

## Short game sound effects

`short_sfx.py` is a second, PROXY-only benchmark for files under 1.5 s: twelve game queries (footstep on gravel, sword swing, coin pickup, UI button click, gunshot, laser shot, jump, punch impact, glass breaking, explosion, magic spell, notification), each with a file-name regex as truth. `search` ranks every short file in each pack-hashed DEV/TEST split by name only, audio only, labels only and fused, reporting AP and P@10. `labels` asks how the AudioSet tagger hears short sounds under twelve input conditions (silence padding to 1, 2 or 10 s; sound at the start or centred; raw or silence-trimmed): a hit is a query label in the top 5. The condition with the most DEV hits is compared with today's (raw, start, 1 s) by an exact paired McNemar test on TEST; a change is adopted only when TEST favours it at p < 0.05. Some queries have no TEST files after the pack split: read the per-query counts in the JSON before trusting a macro number.

```powershell
$env:AUDIO_LIBRARY = 'D:\MySounds'
python .\short_sfx.py search --out short_sfx_search.json
python .\short_sfx.py labels --per-query 120 --out short_sfx_labels.json
```

## Plug in another ranking method

Supply CSV columns `query_id,relative_path,rank` (positive integer rank, unique per query); `score` is optional and ignored for ordering. A full-library ranking is accepted; paths outside a split's known benchmark rows are filtered out. Duplicate paths are collapsed at their first rank. An absent query or short result list is evaluated as incomplete and its coverage is reported.

Alternatively:

```powershell
python .\bench_eval.py run --callable .\my_method.py:rank --mode reviewed
```

A callable `rank(query_dict, candidate_rows)` returns relative path strings (or dicts with `relative_path`) in best-first order. It receives the fixed query and candidate `relative_path` / `duration_seconds` only; truth, notes, splits and nomination strata are withheld. The plugin must not read benchmark truth from disk or learn from test data. A plugin is trusted local Python, not a sandbox. Freeze the method, queries and operating threshold before test evaluation. The fused-search benchmark is a separate diagnostic; these results do not establish listener-reviewed accuracy.

For classification, add a **full-score** CSV with `relative_path,tag,score` and `--tag-scores .\scores.csv --threshold 0.5`. Scores may use any finite scale; the operating threshold must match that scale. Convert model outputs through the named vocabulary mapping, with an explicit aggregation rule of your own. Supply every reviewed path for each tag being assessed. Incomplete per-tag score coverage yields `incomplete_scores`, not invented zero scores. Truncated AST top_labels cannot substitute for full vectors. Unsupported studio-only targets should be left incomplete and will not enter classification summaries. Proxy mode deliberately does not turn query regexes into an exhaustive 89-tag classification truth table.

## Metric definitions

Dev and test are evaluated separately. Uncertain/unreviewed rows are removed from both candidate ranks and relevance (condensed evaluation); report the judged coverage, since partial review can bias scores. This is a closed-pilot evaluation, not full-library human accuracy.

- P@10 and P@50 divide by **10 and 50**, including when fewer results are returned. Missing result slots are misses.
- AP is sum of precision at each retrieved relevant rank divided by **all known relevant** items in that split, including relevant items not returned. Queries with zero positive support get AP `null`/NA, excluded from mAP; their P@k stays zero when the judged candidate set is nonempty. Empty judged sets give NA.
- Retrieval macro is the mean over defined query scores. Micro P@k pools hits and k result slots across queries. `AP_support_weighted` is weighted by positive support; it is explicitly not a separate universal micro-AP definition.
- Per-tag precision/recall/F1 show TP/FP/FN and positive support at `score >= threshold`. Undefined zero denominators are NA. Macro averages defined values; `macro_supported` further restricts to positive-support tags. Micro pools TP/FP/FN across complete tags. No calibration or confidence intervals are claimed.

Proxy regexes match **basename only** and optional excluded basename terms, plus duration limits. They do not prove voice absence, source identity, quietness, mood or suitability. Labels and filenames were also used to nominate the pilot, so selection bias remains. Do not select a winning model from proxy numbers.

## CED-base adopted (measured 11 October 2026)

CED-base replaced AST as the default AudioSet tagger after a paired comparison on **513 filename-sampled clips**. On identical opening excerpts, CED-base alone matched the expected family on **59 clips**, while AST alone matched it on **21** (**paired McNemar p < 0.0001**). The target was a filename-derived family in the top five labels, so this measures agreement with names, not listener-reviewed accuracy. CED-base took approximately half the time per clip in that trial. `tagger_trial.py` retains the paired test and saves full score vectors for reproducibility.

Every newly tagged row records its tagger. Set `AUDIO_TAGGER=ast` to use the previous model. The CED model code is pinned to the reviewed revision in `scripts/audio_tag.py`; its checkpoint and model code use Apache-2.0. CLAP fingerprints remain a separate evidence source. Short-file fingerprints use silence padding rather than repeated audio.

The later short-effect input experiment did not justify a label-input change: the chosen DEV condition produced 9 new-only versus 10 old-only TEST hits (p = 1.0; measured 12 October 2026). It was not adopted. A listener's judgement still takes precedence over names, labels and similarity.

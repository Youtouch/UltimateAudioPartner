# Find and check sound categories

UAP combines existing audio recognition, measured descriptors and filename references to help you browse a sound library. Each source is shown separately. A filename is a publisher hint; a model output is recognition evidence; your saved listening answer is a judgement. None alone proves that every matching sound has been found.

## Start

```text
python skills/audio-designer/scripts/audio_tag.py tag --root MY_LIBRARY
python skills/audio-designer/scripts/audio_describe.py describe --root MY_LIBRARY
python skills/audio-designer/tagging/recall_tags.py build --root MY_LIBRARY
python skills/audio-designer/scripts/sound_browser.py --root MY_LIBRARY
```

The category catalogue uses the official UCS taxonomy, authored aliases and correspondences to existing model labels. It does not fit classifiers or download models. It also works without model files: filenames supply explicitly marked candidates. The tagger separately downloads its runtime models when needed; see NOTICE for licences.

Exact CatIDs, parent categories and families such as click, footstep, impact and notification resolve to related subcategories. Coin/pickup and button/UI select include UIMisc as an approximation of game intent. Material and character words refine sorting using measured descriptors. Category queries return every source candidate; `--top` limits ordinary description searches only. Personal aliases can be saved with `tagging/ucs.py alias --root MY_LIBRARY --word blip --catid UIClick`.

## Read the evidence

Candidates remain **likely** and **uncalibrated**. Model similarity is not probability. Agreement between sources is not a measured listening precision. An absent model label does not show that a sound is absent. Short sounds, especially below one second, are difficult to recognize; game intent such as pickup or laser may have no equivalent AudioSet label. Later events beyond the tagger's analysis window can be missed.

The catalogue records source size and modification time. Changed metadata or audio makes evidence stale: retag and build a new catalogue. Original audio is never rewritten. A new catalogue is an immutable sidecar; use a new output name when rebuilding.

## Check by listening

Choose a UCS category and press **Check this category**. Use Y/N or the buttons to answer whether each sound belongs. The check offers up to twenty source candidates, preferring source disagreement. Older catalogues restrict samples to their training packs. Answers save locally in AUDIO_VERDICTS.json, separately from likes and rejects. Its answer rate and Wilson interval describe only the targeted examples you answered, not library-wide precision or recall. Listening is still needed for uncertain sources and creative fit.

The demo uses generated audio and no model cache. Its categories show filename evidence only and cannot support precision claims.

Read-only audio folders use a separate user metadata cache automatically when tagging with the default output. The browser finds that cache; `--metadata-root` also selects an explicit sidecar folder. Unreadable, digital-silence and too-short rows stay visible without model labels.

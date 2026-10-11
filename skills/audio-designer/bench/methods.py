"""Search methods for bench_eval.py's --callable: each ranks the benchmark candidates for one query.

    AUDIO_LIBRARY=D:/MySounds python bench_eval.py run --mode proxy --callable methods.py:fused --out fused.json
    AUDIO_LIBRARY=D:/MySounds python bench_eval.py run --mode proxy --callable methods.py:audio_only --out audio_only.json
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audio_search import Index  # noqa: E402

ROOT = os.environ.get('AUDIO_LIBRARY', '.')  # the library folder holding AUDIO_TAGS.csv
_index = None


def _setup(candidate_rows):
    global _index
    if _index is None:
        _index = Index(ROOT)
    wanted = {r['relative_path'] for r in candidate_rows}
    return np.array([p in wanted for p in _index.paths])


def fused(query, candidate_rows):
    """Audio + name + label (confirming), reciprocal rank fusion: what the browser runs."""
    keep = _setup(candidate_rows)
    results, _ = _index.find(query['query'], top=int(keep.sum()), keep=keep)
    return [r['path'] for r in results]


def audio_only(query, candidate_rows):
    """The audio source alone, with the query and 'the sound of <query>' averaged."""
    keep = _setup(candidate_rows)
    scores, _, _ = _index.sources(query['query'])
    s = np.where(keep, scores['audio'], -np.inf)
    return [_index.paths[i] for i in np.argsort(-s)[:int(keep.sum())]]


def name_only(query, candidate_rows):
    keep = _setup(candidate_rows)
    scores, _, _ = _index.sources(query['query'])
    s = np.where(keep, scores['name'], -np.inf)
    return [_index.paths[i] for i in np.argsort(-s)[:int(keep.sum())]]

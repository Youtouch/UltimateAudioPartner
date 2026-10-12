"""Explicit, model-free metadata for the generated demo; ordinary libraries never use it."""
import json
from pathlib import Path

import numpy as np

SCHEMA = 'uap-synthetic-demo-v1'
DIM = 512


def read_demo(root):
    """A complete, explicit provenance marker opts only this synthetic library into demo behavior."""
    try:
        marker = json.loads((Path(root) / 'UAP_DEMO.json').read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(marker, dict) or marker.get('schema') != SCHEMA:
        return None
    if not all(isinstance(marker.get(k), str) and marker[k] for k in ('labels', 'search', 'similarity')):
        return None
    return marker


def spectrum_vector(samples, sample_rate):
    """Measured, gain-invariant average spectrum. This is deliberately not a learned fingerprint."""
    y = np.asarray(samples, dtype=np.float64).reshape(-1)
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    # 512 bands across the common 0--8 kHz range; load_audio resamples the demo to 16 kHz.
    spectra = [np.abs(np.fft.rfft(y[start:start + 2048] * np.hanning(2048))) ** 2
               for start in range(0, max(1, len(y) - 2048 + 1), 1024)]
    spectrum = np.mean(spectra, axis=0)
    x = np.fft.rfftfreq(2048, 1 / sample_rate)
    spectrum = np.interp(np.linspace(0, 8000, DIM), x, spectrum)
    vector = np.sqrt(np.maximum(spectrum, 0))
    norm = np.linalg.norm(vector)
    return (vector / norm if norm else np.eye(1, DIM)[0]).astype('<f4')


def name_sources(index, query, tokens, stem):
    """Only lexical evidence: generated tags and names are one source, never recognition agreement."""
    wanted = {stem(w) for w in tokens(query)}
    scores = np.array([sum(w in {stem(t) for t in name.split()} for w in wanted)
                       for name in index.names], dtype=float)
    # Zero scores are excluded from fusion rather than fabricating a match for every demo file.
    return {'name': np.where(scores > 0, scores, -np.inf)}, {}, {'name': 1.0}

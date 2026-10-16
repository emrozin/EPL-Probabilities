"""Blending the model with the betting market: the model's probabilities and the market's,
combined on the log-odds scale with a weight chosen on past seasons.

For two outcomes this is a weighted average of log-odds:
    logit(blend) = w * logit(model) + (1 - w) * logit(market)
and for three (home, draw, away) the same idea, a weighted geometric mean rescaled to add up to 1.
w = 1 is the model alone, w = 0 the market alone. The weight is chosen on the tuning seasons and
tested on the held-out ones (nfl/blend.py, epl/blend.py), and saved next to them as blend.json.
"""

import json
from pathlib import Path

import numpy as np

WEIGHTS = np.round(np.arange(0, 1.0001, 0.05), 2)


def pool(model, market, w: float) -> np.ndarray:
    """Blend probabilities. model, market: arrays of shape (n, outcomes), each row adding up to 1."""
    model = np.clip(np.asarray(model, dtype=float), 1e-6, 1)
    market = np.clip(np.asarray(market, dtype=float), 1e-6, 1)
    logp = w * np.log(model) + (1 - w) * np.log(market)
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def log_loss(p, outcome) -> float:
    """Average -log(probability of what happened). outcome: index of the actual outcome per row."""
    p = np.asarray(p)
    return float(-np.log(np.clip(p[np.arange(len(p)), outcome], 1e-12, 1)).mean())


def tune(model, market, outcome) -> tuple[float, list[tuple[float, float]]]:
    """The weight with the lowest log loss, and the log loss at every weight tried."""
    scores = [(float(w), log_loss(pool(model, market, w), outcome)) for w in WEIGHTS]
    return min(scores, key=lambda s: s[1])[0], scores


def save(path: Path, info: dict) -> None:
    path.write_text(json.dumps(info, indent=1))


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None

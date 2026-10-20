"""Shared helpers for Chapter 1. Plain Python + NumPy only.

Everything here is small on purpose: read it once, then you can read every
script in this folder without surprises.
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "outputs")
DEFAULT_TEXT = os.path.join(REPO, "data", "raw", "tinyshakespeare.txt")


# --------------------------------------------------------------------------- #
# Information theory (natural log, so everything is in nats)
# --------------------------------------------------------------------------- #
def entropy(p: np.ndarray) -> float:
    """H(p) = -sum p ln p  (0 ln 0 is treated as 0)."""
    p = np.asarray(p, dtype=float)
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum()) + 0.0   # + 0.0 turns -0.0 into 0.0


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """H(p, q) = -sum p ln q."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    nz = p > 0
    return float(-(p[nz] * np.log(q[nz])).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) = sum p ln(p/q)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / q[nz])).sum())


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (subtract the max before exponentiating)."""
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def perplexity(loss_nats: float) -> float:
    return math.exp(loss_nats)


# --------------------------------------------------------------------------- #
# Text loading
# --------------------------------------------------------------------------- #
def synthetic_text(n_chars: int = 400_000, seed: int = 0) -> str:
    """Language-*shaped* fallback text (Zipf-distributed words) for offline runs.

    It is NOT real language, so the numbers will differ from the blog post,
    but every script still runs and every identity still holds.
    """
    rng = random.Random(seed)
    words = ("the of and to in a is that it was for on are with as his they be at "
             "one have this from or had by not word but what some we can out other "
             "were all there when up use your how said an each she which do their "
             "time if will way about many then them write would like so these her "
             "long make thing see him two has look more day could go come did").split()
    weights = [1.0 / (i + 1) for i in range(len(words))]
    out: List[str] = []
    size = 0
    while size < n_chars:
        line = " ".join(rng.choices(words, weights, k=rng.randint(5, 12)))
        line = line.capitalize() + rng.choice([".", ",", "?", "!"])
        out.append(line)
        size += len(line) + 1
    return "\n".join(out) + "\n"


def load_text(path: Optional[str] = None, max_chars: Optional[int] = None) -> str:
    """Load a plain-text file. Falls back to downloaded Tiny Shakespeare,
    then to synthetic text (with a warning) so scripts never crash offline."""
    candidates = [path] if path else [DEFAULT_TEXT]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            return text[:max_chars] if max_chars else text
    if path:
        raise FileNotFoundError(f"--text file not found: {path}")
    print("[note] data/raw/tinyshakespeare.txt not found. Run `python data/get_data.py` "
          "or pass --text. Using SYNTHETIC text, so numbers will differ from the post.")
    return synthetic_text(max_chars or 400_000)


def split_train_heldout(text: str, heldout_frac: float = 0.1) -> Tuple[str, str]:
    """Split on a line boundary: first 90% train, last 10% held-out."""
    cut = int(len(text) * (1 - heldout_frac))
    nl = text.find("\n", cut)
    cut = nl + 1 if nl != -1 else cut
    return text[:cut], text[cut:]


# --------------------------------------------------------------------------- #
# Spam injection (used by 1.2.4 and 1.3.5)
# --------------------------------------------------------------------------- #
SPAM_LINES = [
    "CLICK HERE to BUY FREE PRIZE now www.win-big-today.example",
    "FREE FREE FREE!!! BUY NOW and CLICK www.cheap-deals.example",
    "WINNER! CLICK to claim your FREE PRIZE at www.prize-zone.example",
    "BUY cheap pills ONLINE, CLICK www.best-offers.example FREE shipping",
]


def inject_spam(text: str, every: int = 4, seed: int = 0,
                lines: Optional[List[str]] = None) -> Tuple[str, float]:
    """Insert one spam line after every `every`-th non-empty line.

    Returns (polluted_text, spam_share) where spam_share is the fraction of
    CHARACTERS in the result that belong to spam lines (this is our epsilon).
    """
    rng = random.Random(seed)
    lines = lines or SPAM_LINES
    out: List[str] = []
    spam_chars = 0
    count = 0
    for line in text.split("\n"):
        out.append(line)
        if line.strip():
            count += 1
            if count % every == 0:
                s = rng.choice(lines)
                out.append(s)
                spam_chars += len(s) + 1
    polluted = "\n".join(out)
    return polluted, spam_chars / max(len(polluted), 1)


def is_spam_line(line: str) -> bool:
    """A deliberately simple 'filter': exact match against known spam templates."""
    return line.strip() in SPAM_LINES


def remove_spam(text: str) -> str:
    return "\n".join(l for l in text.split("\n") if not is_spam_line(l))


# --------------------------------------------------------------------------- #
# A tiny character n-gram language model (pure counts, add-alpha smoothing)
# --------------------------------------------------------------------------- #
class CharNgram:
    """Predict the next character from the previous `order` characters.

    P(c | context) = (count(context, c) + alpha) / (count(context) + alpha * V)

    This is the 'frequency table' model from 1.2.2, plus smoothing so unseen
    events do not get probability zero.
    """

    def __init__(self, order: int = 3, alpha: float = 0.05, vocab: Optional[Iterable[str]] = None):
        self.order = order
        self.alpha = alpha
        self.vocab = set(vocab) if vocab is not None else None
        self.counts: Dict[str, Dict[str, int]] = {}
        self.totals: Dict[str, int] = {}

    def fit(self, text: str) -> "CharNgram":
        if self.vocab is None:
            self.vocab = set(text)
        k = self.order
        counts: Dict[str, Dict[str, int]] = {}
        for i in range(k, len(text)):
            ctx, ch = text[i - k:i], text[i]
            d = counts.get(ctx)
            if d is None:
                d = counts[ctx] = {}
            d[ch] = d.get(ch, 0) + 1
        self.counts = counts
        self.totals = {c: sum(d.values()) for c, d in counts.items()}
        return self

    @property
    def V(self) -> int:
        return len(self.vocab)  # type: ignore[arg-type]

    def prob(self, ctx: str, ch: str) -> float:
        d = self.counts.get(ctx)
        n_ctx = self.totals.get(ctx, 0)
        n = d.get(ch, 0) if d else 0
        return (n + self.alpha) / (n_ctx + self.alpha * self.V)

    def loss(self, text: str) -> float:
        """Average negative log-probability per character, in nats."""
        k = self.order
        total, n = 0.0, 0
        for i in range(k, len(text)):
            ch = text[i]
            if ch not in self.vocab:  # type: ignore[operator]
                continue
            total -= math.log(self.prob(text[i - k:i], ch))
            n += 1
        return total / max(n, 1)

    def sample(self, n: int = 300, seed: int = 0, prompt: Optional[str] = None) -> str:
        """Sample text. Uses the observed counts when the context was seen in training
        (so output looks like the data) and falls back to uniform otherwise."""
        rng = random.Random(seed)
        vocab = sorted(self.vocab)  # type: ignore[arg-type]
        text = prompt or rng.choice(list(self.counts))
        while len(text) < n:
            d = self.counts.get(text[-self.order:] if self.order else "")
            if d:
                chars, weights = zip(*d.items())
                text += rng.choices(chars, weights)[0]
            else:
                text += rng.choice(vocab)
        return text


# --------------------------------------------------------------------------- #
# Plot helper (matplotlib is optional)
# --------------------------------------------------------------------------- #
def save_plot(fig_builder, filename: str) -> Optional[str]:
    """Call fig_builder(plt) and save to chapter-01/outputs/<filename>.
    Silently skips (with a note) when matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[note] matplotlib not installed, skipping plot {filename}")
        return None
    os.makedirs(OUT_DIR, exist_ok=True)
    fig_builder(plt)
    path = os.path.join(OUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"[plot] saved {os.path.relpath(path, REPO)}")
    return path


def header(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


# --------------------------------------------------------------------------- #
# Scaling-law constants   L(N, D) = E + A / N^alpha + B / D^beta
# --------------------------------------------------------------------------- #
SCALING_CONSTANTS = {
    # Hoffmann et al. 2022, Approach 3 (rounded as in the blog post; the paper's
    # unrounded exponents are alpha = 0.3392, beta = 0.2849, E = 1.693).
    "hoffmann": dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28),
    # Besiroglu et al. 2024 (Epoch AI) replication attempt of the same fit.
    "epoch": dict(E=1.817, A=482.0, B=2085.43, alpha=0.3478, beta=0.3658),
}


def scaling_loss(N: float, D: float, c: str = "hoffmann") -> float:
    k = SCALING_CONSTANTS[c]
    return k["E"] + k["A"] / N ** k["alpha"] + k["B"] / D ** k["beta"]


def scaling_terms(N: float, D: float, c: str = "hoffmann") -> Tuple[float, float, float]:
    k = SCALING_CONSTANTS[c]
    return k["E"], k["A"] / N ** k["alpha"], k["B"] / D ** k["beta"]

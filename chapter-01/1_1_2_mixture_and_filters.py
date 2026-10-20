"""1.1.2  The pipeline as three operations on a distribution.

    Mixing        p_mix(x)      = sum_k w_k * p_k(x)
    Filtering     p_filtered(x) = p(x) * f(x) / Z      (Z = keep rate)
    Deduplication sets every document's multiplicity m(x) back to 1

It also reproduces the two worked examples: filters multiply, and
tokens-seen vs tokens-owned (epochs).

    python chapter-01/1_1_2_mixture_and_filters.py
"""
from collections import Counter

from common import header

# ---- a toy corpus: (source, text) -----------------------------------------
FOOTER = "subscribe to our newsletter | privacy | terms"
CORPUS = (
    [("web", f"web article number {i} about topic {i % 5}") for i in range(40)]
    + [("web", FOOTER)] * 30                      # boilerplate repeated 30x
    + [("books", f"book passage {i} a carefully edited sentence") for i in range(20)]
    + [("code", f"def func_{i}(): return {i}") for i in range(10)]
)


def check_weights(source_weights):
    """Mixing weights w_k are design decisions, but they must sum to 1."""
    assert abs(sum(source_weights.values()) - 1) < 1e-9, "weights must sum to 1"
    return source_weights


def shares(docs):
    c = Counter(src for src, _ in docs)
    n = sum(c.values())
    return {k: v / n for k, v in c.items()}


def apply_filter(docs, f):
    """Hard filter f(doc) -> 0/1. Returns (kept_docs, Z) where Z is the keep rate."""
    kept = [d for d in docs if f(d)]
    return kept, len(kept) / len(docs)


def dedupe(docs):
    seen, out = set(), []
    for d in docs:
        if d[1] not in seen:
            seen.add(d[1])
            out.append(d)
    return out


def print_shares(title, docs):
    header(title + f"  ({len(docs)} docs)")
    for k, v in sorted(shares(docs).items()):
        print(f"  {k:<6} {v:6.1%}")
    footer_share = sum(1 for _, t in docs if t == FOOTER) / len(docs)
    print(f"  boilerplate footer share: {footer_share:.1%}")


def main():
    print_shares("Raw corpus", CORPUS)

    # Operation 3: deduplication
    deduped = dedupe(CORPUS)
    print_shares("After deduplication", deduped)
    m_footer = sum(1 for _, t in CORPUS if t == FOOTER)
    print(f"  the footer was {m_footer}x over-represented before dedup")

    # Operation 2: filtering (drop short code snippets)
    filtered, Z = apply_filter(deduped, lambda d: d[0] != "code")
    print_shares("After filter (drop 'code')", filtered)
    print(f"  keep rate Z = {Z:.3f}")

    # Operation 1: mixing weights are design decisions
    header("Mixing weights are design decisions (Llama 3 style)")
    w = {"general": 0.50, "math/reasoning": 0.25, "code": 0.17, "multilingual": 0.08}
    check_weights(w)
    for k, v in w.items():
        print(f"  {k:<16} {v:.0%}")
    print(f"  sum = {sum(w.values()):.2f}")

    # Worked example 1: filters multiply
    header("Worked example 1: filters multiply")
    rates = {"URL": 0.90, "language": 0.60, "quality": 0.50, "dedup": 0.50}
    z = 1.0
    for name, r in rates.items():
        z *= r
        print(f"  after {name:<9} keep rate {r:.0%}  -> cumulative Z = {z:.4f}")
    print(f"  to end with 1B clean tokens, start with {1e9 / z / 1e9:.1f}B raw tokens")

    # Worked example 2: epochs
    header("Worked example 2: tokens seen vs tokens owned")
    D, w_k, U_k = 15e12, 0.0001, 200e6
    seen = w_k * D
    print(f"  tokens drawn  = w_k * D = {w_k} * 15T = {seen / 1e9:.1f}B")
    print(f"  unique tokens = {U_k / 1e6:.0f}M")
    print(f"  epochs        = {seen / U_k:.1f}   (more than ~4 epochs has fast-falling returns)")


if __name__ == "__main__":
    main()

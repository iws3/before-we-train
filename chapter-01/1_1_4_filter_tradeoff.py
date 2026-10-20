"""1.1.4  Quantity, quality, diversity: measured.

Keep documents with quality score q(x) >= tau, then measure

    Quantity   n(tau)  = number of documents kept
    Quality    mean q  = average score of the kept documents
    Diversity  H(tau)  = entropy of the group shares in the kept set (nats)

Reproduces the 10-document table from the post, then sweeps tau and plots.

    python chapter-01/1_1_4_filter_tradeoff.py
"""
import numpy as np

from common import entropy, header, save_plot

# (group, quality score from an English-trained classifier)
DOCS = (
    [("English", s) for s in (0.90, 0.85, 0.80, 0.75, 0.70, 0.30)]
    + [("Forum", s) for s in (0.60, 0.20)]
    + [("Yoruba", s) for s in (0.20, 0.15)]
)
GROUPS = ["English", "Forum", "Yoruba"]


def measure(docs, tau):
    kept = [(g, q) for g, q in docs if q >= tau - 1e-9]   # tolerance for float thresholds
    if not kept:
        return 0, float("nan"), 0.0, {g: 0.0 for g in GROUPS}
    n = len(kept)
    mean_q = sum(q for _, q in kept) / n
    shares = {g: sum(1 for k, _ in kept if k == g) / n for g in GROUPS}
    return n, mean_q, entropy(np.array(list(shares.values()))), shares


def main():
    header("Before / after at tau = 0.5")
    n0, q0, h0, s0 = measure(DOCS, 0.0)
    n1, q1, h1, s1 = measure(DOCS, 0.5)
    print(f"  {'':<10}{'before':>12}{'after':>12}")
    print(f"  {'quantity':<10}{n0:>12}{n1:>12}")
    print(f"  {'quality':<10}{q0:>12.3f}{q1:>12.3f}")
    print(f"  {'diversity':<10}{h0:>12.3f}{h1:>12.3f}   (nats)")
    print("  shares before:", {g: f"{v:.0%}" for g, v in s0.items()})
    print("  shares after :", {g: f"{v:.0%}" for g, v in s1.items()})
    print(f"\n  quality  x{q1 / q0:.2f}   |   diversity x{h1 / h0:.2f}")
    print("  In this toy data the Yoruba pages are not bad. The classifier never learned")
    print("  what good Yoruba looks like, so the filter erased the language.")

    header("Sweep tau from 0 to 1")
    taus = np.linspace(0, 1, 21)
    rows = [measure(DOCS, t) for t in taus]
    print(f"  {'tau':>5} {'n':>3} {'quality':>8} {'diversity':>10}")
    for t, (n, q, h, _) in zip(taus, rows):
        qs = f"{q:8.3f}" if n else "     n/a"
        print(f"  {t:5.2f} {n:>3} {qs} {h:10.3f}")

    def build(plt):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(taus, [r[0] / len(DOCS) for r in rows], label="quantity (share kept)")
        ax.plot(taus, [r[1] if r[0] else np.nan for r in rows], label="quality (mean q)")
        ax.plot(taus, [r[2] / np.log(len(GROUPS)) for r in rows], label="diversity (H / ln 3)")
        ax.set_xlabel("threshold tau")
        ax.set_title("Raising the threshold buys quality and pays in quantity and diversity")
        ax.legend()
    save_plot(build, "1_1_4_filter_tradeoff.png")


if __name__ == "__main__":
    main()

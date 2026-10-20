"""1.3.5  A mini scaling experiment you can run on a laptop.

You cannot afford 30 million GPU-hours, but you CAN measure the data term
B / D^beta for your own data:

  1. train a tiny model on growing slices of the corpus (1/64 ... 1/1)
  2. record held-out loss against the number of training characters D
  3. fit  L = E + B / D^beta  (grid over E, straight line on log-log axes)
  4. repeat for a second dataset version and compare

Here the two versions are  (a) text polluted with spam  and  (b) the same text after
a simple spam filter. Swap in any two versions of YOUR pipeline.

    python chapter-01/1_3_5_mini_scaling_experiment.py
    python chapter-01/1_3_5_mini_scaling_experiment.py --text my_language.txt
"""
import argparse
import random

import numpy as np

from common import (CharNgram, header, inject_spam, is_spam_line, load_text,
                    save_plot, split_train_heldout)


def fit_power_law(D, L):
    """Fit L = E + B / D^beta. Grid over E; for each E, least squares on the log-log line."""
    D, L = np.asarray(D, float), np.asarray(L, float)
    best = None
    for E in np.linspace(0, L.min() - 1e-3, 400):
        y = np.log(L - E)
        slope, intercept = np.polyfit(np.log(D), y, 1)
        resid = float(((y - (slope * np.log(D) + intercept)) ** 2).sum())
        if best is None or resid < best[0]:
            best = (resid, E, float(np.exp(intercept)), float(-slope))
    _, E, B, beta = best
    return E, B, beta


def learning_curve(train_lines, heldout, vocab, fractions, order, alpha):
    """Train on the first `fraction` of an already-shuffled list of lines."""
    Ds, Ls = [], []
    for f in fractions:
        part = "\n".join(train_lines[: int(len(train_lines) * f)])
        m = CharNgram(order, alpha, vocab).fit(part)
        Ds.append(len(part))
        Ls.append(m.loss(heldout))
    return Ds, Ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--every", type=int, default=4)
    args = ap.parse_args()

    text = load_text(args.text)
    train, heldout = split_train_heldout(text)
    polluted, eps = inject_spam(train, every=args.every)
    vocab = set(text) | set(polluted)
    # Shuffle lines so every slice is a random sample (otherwise later slices drift
    # toward whatever play comes next and the curve is not smooth).
    polluted_lines = polluted.split("\n")
    random.Random(0).shuffle(polluted_lines)
    filtered_lines = [l for l in polluted_lines if not is_spam_line(l)]
    fractions = [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]

    versions = {"polluted": polluted_lines, "filtered": filtered_lines}
    curves = {}
    for name, t in versions.items():
        curves[name] = learning_curve(t, heldout, vocab, fractions, args.order, 0.05)

    header(f"Held-out loss (nats/char) vs training characters, order-{args.order} model")
    print(f"  spam share of the polluted data: {eps:.1%}\n")
    print(f"  {'slice':>6} | {'polluted: D':>12} {'loss':>7} | {'filtered: D':>12} {'loss':>7}")
    for i, f in enumerate(fractions):
        Dp, Lp = curves["polluted"][0][i], curves["polluted"][1][i]
        Df, Lf = curves["filtered"][0][i], curves["filtered"][1][i]
        print(f"  {f:>6.3f} | {Dp:>12,} {Lp:>7.4f} | {Df:>12,} {Lf:>7.4f}")

    header("Fit  L = E + B / D^beta")
    fits = {}
    for name, (D, L) in curves.items():
        E, B, beta = fit_power_law(D, L)
        fits[name] = (E, B, beta)
        print(f"  {name:<9} E = {E:.3f}   B = {B:.3g}   beta = {beta:.3f}")
    print("\n  beta is the slope on log-log axes: how fast more data helps this pipeline.")
    print("  With 7 points and a toy model, treat the fit as a demonstration of the method,")
    print("  not a measurement. The more reliable signal is the direct comparison below.")

    header("Did the filter help? (same held-out text, full training set)")
    Lp, Lf = curves["polluted"][1][-1], curves["filtered"][1][-1]
    print(f"  polluted : {Lp:.4f}\n  filtered : {Lf:.4f}\n  change   : {Lf - Lp:+.4f} nats/char "
          f"({'better' if Lf < Lp else 'worse'}), "
          f"with {1 - len(filtered_lines) / len(polluted_lines):.1%} fewer lines")

    def build(plt):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for name, (D, L) in curves.items():
            ax.plot(D, L, "o-", label=name)
            E, B, beta = fits[name]
            xs = np.logspace(np.log10(min(D)), np.log10(max(D)), 50)
            ax.plot(xs, E + B / xs ** beta, "--", alpha=0.5)
        ax.set(xscale="log", xlabel="training characters D", ylabel="held-out loss (nats/char)",
               title="Data scaling curve, two dataset versions")
        ax.legend()
    save_plot(build, "1_3_5_mini_scaling.png")


if __name__ == "__main__":
    main()

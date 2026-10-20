"""1.3.3  Scaling laws with real numbers.

    L(N, D) = E + A / N^alpha + B / D^beta
              floor  model term   data term

Reproduces the two rows of the table in the post, the "doubling the data" factor,
and exercise 4 (how many tokens to push the data term under 0.05). Optionally
plots loss curves.

    python chapter-01/1_3_3_chinchilla_calculator.py
    python chapter-01/1_3_3_chinchilla_calculator.py --constants epoch
"""
import argparse

import numpy as np

from common import SCALING_CONSTANTS, header, save_plot, scaling_loss, scaling_terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constants", default="hoffmann", choices=list(SCALING_CONSTANTS))
    args = ap.parse_args()
    c = args.constants
    k = SCALING_CONSTANTS[c]

    header(f"Constants ({c})")
    print("  " + "  ".join(f"{n} = {v}" for n, v in k.items()))
    print("  These belong to the paper's own dataset and tokenizer: the SHAPE transfers,")
    print("  the absolute losses do not.")

    header("Two real model sizes")
    print(f"  {'model':<22} {'A/N^a':>8} {'B/D^b':>8} {'L total':>8}")
    for name, N, D in [("Llama 3 8B, 15T", 8e9, 15e12), ("Llama 3 405B, 15.6T", 405e9, 15.6e12)]:
        E, mt, dt = scaling_terms(N, D, c)
        print(f"  {name:<22} {mt:>8.3f} {dt:>8.3f} {E + mt + dt:>8.3f}")
    print("\n  At 405B the data term is the bigger of the two reducible terms:")
    print("  data is what still holds the loss up.")

    header("What doubling the data buys")
    f = 2 ** (-k["beta"])
    print(f"  2^(-beta) = {f:.3f}  -> the data term shrinks by {1 - f:.0%}, and E never moves")

    header("Exercise 4: tokens needed to push the 8B model's data term below 0.05")
    D_needed = (k["B"] / 0.05) ** (1 / k["beta"])
    print(f"  D = (B / 0.05)^(1/beta) = {D_needed:.3e} tokens  ({D_needed / 15e12:.1f}x Llama 3's 15T)")

    def build(plt):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        Ds = np.logspace(9, 14, 60)
        for N in (1e8, 1e9, 8e9, 70e9, 405e9):
            axes[0].plot(Ds, [scaling_loss(N, d, c) for d in Ds], label=f"N = {N:.0e}")
        axes[0].set(xscale="log", xlabel="training tokens D", ylabel="loss",
                    title="More data lowers loss, with diminishing returns")
        axes[0].axhline(k["E"], ls="--", c="gray")
        axes[0].legend()
        Ns = np.logspace(7, 12, 60)
        for D in (1e10, 1e11, 1e12, 15e12):
            axes[1].plot(Ns, [scaling_loss(n, D, c) for n in Ns], label=f"D = {D:.0e}")
        axes[1].set(xscale="log", xlabel="parameters N", title="Bigger models help, until data is the limit")
        axes[1].axhline(k["E"], ls="--", c="gray")
        axes[1].legend()
    save_plot(build, f"1_3_3_loss_curves_{c}.png")


if __name__ == "__main__":
    main()

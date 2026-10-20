"""1.2.5  The ln(V) sanity check.

A model that knows nothing assigns 1/V to every token, so its cross-entropy is
exactly ln V on ANY data. Your first logged training loss should sit right there.

Part A: ln V for common tokenizers.
Part B: a randomly initialized model, at three initial weight scales, to show the
        three failure regimes (too confident / correct / too small to learn).

    python chapter-01/1_2_5_ln_vocab_check.py
"""
import math

import numpy as np

from common import header, softmax

VOCABS = {"Llama 2": 32_000, "GPT-2": 50_257, "Llama 3": 128_256}


def starting_loss(V, init_std, n_ctx=64, n_examples=2000, seed=0):
    """Average loss of a random 'logits = W[context]' model on random next tokens."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, init_std, size=(n_ctx, V))
    ctx = rng.integers(0, n_ctx, size=n_examples)
    y = rng.integers(0, V, size=n_examples)
    p = softmax(W[ctx])
    return float(-np.log(p[np.arange(n_examples), y]).mean())


def main():
    header("Part A: expected starting loss = ln V")
    print(f"  {'tokenizer':<10} {'V':>8} {'ln V':>8}")
    for name, V in VOCABS.items():
        print(f"  {name:<10} {V:>8,} {math.log(V):>8.2f}")

    header("Part B: a randomly initialized model (V = 50,257)")
    V = 50_257
    print(f"  expected ln V = {math.log(V):.3f}\n")
    print(f"  {'init std':>9} {'start loss':>11}   reading")
    for std in (0.0, 0.02, 1.0, 5.0):
        L = starting_loss(V, std)
        if L < math.log(V) + 0.05:
            note = "right at ln V: healthy start"
        else:
            note = "FAR ABOVE ln V: overconfident (weights too large)"
        print(f"  {std:>9} {L:>11.3f}   {note}")

    header("How to use it")
    print("  far above ln V : overconfident init (weights too large)")
    print("  far below ln V : suspect label leakage or trivially predictable data (duplication)")
    print("  at ln V, never falling : no usable signal (misaligned inputs/targets, or noise)")


if __name__ == "__main__":
    main()

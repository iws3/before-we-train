"""1.3.1  From tokens to loss to perplexity, and bits per byte.

    L   = -(1/T) * sum_t ln p(x_t | x_<t)
    PPL = e^L = geometric mean of the inverse probabilities
    BPB = (T * L) / (B * ln 2)              (tokenizer-independent)

Also shows why log space is not optional: multiplying 1,000 probabilities underflows.

    python chapter-01/1_3_1_perplexity_and_bpb.py
"""
import math

import numpy as np

from common import header


def loss_from_probs(probs):
    return float(-np.mean(np.log(probs)))


def bits_per_byte(n_tokens, loss_nats, n_bytes):
    return (n_tokens * loss_nats) / (n_bytes * math.log(2))


def main():
    header("Worked example: four tokens")
    probs = np.array([0.5, 0.25, 0.1, 0.8])
    print(f"  {'token':>5} {'p':>6} {'ln p':>9}")
    for i, p in enumerate(probs, 1):
        print(f"  {i:>5} {p:>6.2f} {math.log(p):>9.4f}")
    L = loss_from_probs(probs)
    print(f"\n  sum ln p = {np.log(probs).sum():.4f}     loss L = {L:.4f}")
    print(f"  PPL = e^L                    = {math.exp(L):.4f}")
    print(f"  PPL = (1 / prod p)^(1/T)     = {(1 / probs.prod()) ** (1 / len(probs)):.4f}   (same number)")

    header("Why log space is not optional")
    p, T = 0.1, 1000
    naive = 1.0
    for _ in range(T):
        naive *= p
    print(f"  1,000 tokens at p = 0.1, multiplied directly : {naive}   <- underflow to zero")
    print(f"  smallest positive float64                    : {np.finfo(float).tiny:.3e}")
    print(f"  summing logs instead                         : {T * math.log(p):.1f}")

    header("Bits per byte: compare models across tokenizers")
    tokens, nbytes, L = 1000, 4200, 2.3
    print(f"  {tokens} tokens, {nbytes} bytes, L = {L} nats/token")
    print(f"  BPB = ({tokens} x {L}) / ({nbytes} x ln 2) = {bits_per_byte(tokens, L, nbytes):.3f}")
    print("\n  Same text, a tokenizer that makes 2x more (easier) tokens, per-token loss halves:")
    print(f"  per-token PPL: {math.exp(2.3):.2f} -> {math.exp(1.15):.2f}   (looks better)")
    print(f"  BPB          : {bits_per_byte(1000, 2.3, 4200):.3f} -> {bits_per_byte(2000, 1.15, 4200):.3f}   (identical)")


if __name__ == "__main__":
    main()

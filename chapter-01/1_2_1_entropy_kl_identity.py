"""1.2.1  Entropy, cross-entropy, KL: check the identity numerically.

    H(p, q) = H(p) + KL(p || q)        and        KL >= 0

Draw many random distributions p and q, compute all three quantities, and
verify the identity to machine precision and that KL is never negative.

    python chapter-01/1_2_1_entropy_kl_identity.py
"""
import numpy as np

from common import cross_entropy, entropy, header, kl_divergence


def random_dist(rng, k):
    x = rng.random(k) + 1e-3
    return x / x.sum()


def main():
    rng = np.random.default_rng(0)
    worst_identity_gap, min_kl = 0.0, np.inf
    for _ in range(10_000):
        k = int(rng.integers(2, 50))
        p, q = random_dist(rng, k), random_dist(rng, k)
        h, hx, kl = entropy(p), cross_entropy(p, q), kl_divergence(p, q)
        worst_identity_gap = max(worst_identity_gap, abs(hx - (h + kl)))
        min_kl = min(min_kl, kl)

    header("10,000 random (p, q) pairs")
    print(f"  largest |H(p,q) - (H(p) + KL)| = {worst_identity_gap:.2e}   (machine precision)")
    print(f"  smallest KL seen               = {min_kl:.2e}   (never negative)")

    header("A concrete example")
    p = np.array([0.5, 0.25, 0.25])
    q = np.array([0.4, 0.4, 0.2])
    print(f"  p = {p},  q = {q}")
    print(f"  H(p)      = {entropy(p):.4f}   <- the floor, training cannot move it")
    print(f"  KL(p||q)  = {kl_divergence(p, q):.4f}   <- the only part training can reduce")
    print(f"  H(p,q)    = {cross_entropy(p, q):.4f}   = {entropy(p):.4f} + {kl_divergence(p, q):.4f}")
    print(f"  with q = p: KL = {kl_divergence(p, p):.1e}, so H(p,p) = H(p) = {cross_entropy(p, p):.4f}")


if __name__ == "__main__":
    main()

"""1.3.2  How the data pushes on the model: the softmax gradient.

For logits z, probabilities p = softmax(z) and loss l = -ln p_y:

    dl/dz_i = p_i - 1[i = y]          "what the model predicted minus what happened"

1. Reproduce the 3-token example (logits 2, 1, 0; true token 3).
2. Verify the formula against numerical finite-difference gradients.
3. Show the dataset-averaged gradient goes to zero exactly when the model's
   distribution matches the data's distribution for that context.

    python chapter-01/1_3_2_softmax_gradient_check.py
"""
import numpy as np

from common import header, softmax


def loss(z, y):
    return float(-np.log(softmax(z)[y]))


def analytic_grad(z, y):
    g = softmax(z)
    g[y] -= 1.0
    return g


def numeric_grad(z, y, eps=1e-6):
    g = np.zeros_like(z, dtype=float)
    for i in range(len(z)):
        zp, zm = z.astype(float).copy(), z.astype(float).copy()
        zp[i] += eps
        zm[i] -= eps
        g[i] = (loss(zp, y) - loss(zm, y)) / (2 * eps)
    return g


def main():
    header("Worked example: logits (2, 1, 0), true token = 3")
    z, y = np.array([2.0, 1.0, 0.0]), 2
    p, g = softmax(z), analytic_grad(z, y)
    print(f"  {'token':>5} {'prob':>7} {'gradient':>9}")
    for i in range(3):
        print(f"  {i + 1:>5} {p[i]:>7.3f} {g[i]:>+9.3f}   "
              f"{'<- push score UP hard' if i == y else 'push score down'}")
    print(f"\n  loss = -ln {p[y]:.3f} = {loss(z, y):.2f}      gradient sums to {g.sum():+.1e} (probability is conserved)")

    header("Analytic vs numerical gradient (500 random cases)")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        k = int(rng.integers(2, 20))
        z = rng.normal(0, 3, k)
        y = int(rng.integers(0, k))
        worst = max(worst, np.abs(analytic_grad(z, y) - numeric_grad(z, y)).max())
    print(f"  largest disagreement = {worst:.2e}")

    header("Averaged over the data, the pull vanishes only at the data distribution")
    data_p = np.array([0.70, 0.20, 0.10])            # what the data says follows this context
    spam = 0.10                                       # token 3 plays the role of spam
    z = np.zeros(3)
    print(f"  data says {data_p} (spam token has {spam:.0%})")
    print(f"  {'step':>5} {'q':>26} {'|mean gradient|':>16}")
    for step in range(0, 401):
        q = softmax(z)
        mean_grad = q - data_p                        # E_data[p - onehot] = q - p_data
        if step in (0, 1, 10, 50, 100, 400):
            print(f"  {step:>5} {str(np.round(q, 3)):>26} {np.abs(mean_grad).max():>16.2e}")
        z -= 0.5 * mean_grad
    print("\n  The update keeps pulling until q = p_data. The spam token keeps getting its")
    print(f"  {spam:.0%} share, step after step, for as long as training runs.")


if __name__ == "__main__":
    main()

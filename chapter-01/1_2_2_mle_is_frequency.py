"""1.2.2  Training is maximum likelihood: the optimum is the frequency table.

A model is one categorical distribution over K outcomes, parameterized by logits.
Run gradient descent on the average negative log-likelihood and watch the learned
probabilities converge to n_k / n.

    python chapter-01/1_2_2_mle_is_frequency.py
"""
import numpy as np

from common import cross_entropy, entropy, header, softmax


def main():
    rng = np.random.default_rng(1)
    K = 5
    # a dataset of 2,000 tokens drawn from some true distribution
    true_p = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
    data = rng.choice(K, size=2000, p=true_p)
    counts = np.bincount(data, minlength=K)
    freq = counts / counts.sum()                      # n_k / n  (the claimed optimum)

    z = np.zeros(K)                                   # logits, start at "know nothing"
    lr = 0.5
    header("Gradient descent on the logits")
    print(f"  {'step':>5} {'loss':>9} {'KL to freq':>12}")
    for step in range(0, 301):
        q = softmax(z)
        if step in (0, 1, 5, 20, 50, 100, 200, 300):
            kl = float((freq * np.log(freq / q)).sum())
            print(f"  {step:>5} {cross_entropy(freq, q):>9.5f} {kl:>12.2e}")
        z -= lr * (q - freq)                          # gradient of the mean loss: q - p_hat

    q = softmax(z)
    header("Where it ended up")
    print(f"  {'outcome':>8} {'n_k':>6} {'n_k / n':>9} {'learned q_k':>12}")
    for k in range(K):
        print(f"  {k:>8} {counts[k]:>6} {freq[k]:>9.4f} {q[k]:>12.4f}")
    print(f"\n  max |q_k - n_k/n| = {np.abs(q - freq).max():.2e}")
    print(f"  final loss {cross_entropy(freq, q):.5f}  vs  H(p_hat) = {entropy(freq):.5f}"
          "  (KL has gone to ~0)")
    print("\n  The target was the dataset's frequencies, not the true p:")
    print(f"  true p      = {true_p}")
    print(f"  dataset p^  = {np.round(freq, 4)}")


if __name__ == "__main__":
    main()
